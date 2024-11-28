import argparse
import gzip
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fsspec
import gcsfs
import pandas as pd
import pyarrow.parquet as pq
import ray
import s3fs
from huggingface_hub import hf_hub_download
from tqdm import tqdm


@ray.remote
class StatsActor:
    def __init__(self):
        self.parquet_download_times = []
        self.repo_download_times = []
        self.repos_processed = 0
        self.files_processed = 0
        self.bytes_downloaded = 0
        self.start_time = time.time()

    def add_parquet_time(self, size_bytes: int, duration: float):
        self.parquet_download_times.append((size_bytes, duration))
        self.bytes_downloaded += size_bytes

    def add_repo_time(self, num_files: int, duration: float):
        self.repo_download_times.append((num_files, duration))
        self.repos_processed += 1
        self.files_processed += num_files

    def get_stats(self) -> Dict:
        if not self.parquet_download_times:
            return {}

        total_parquet_bytes = sum(size for size, _ in self.parquet_download_times)
        total_parquet_time = sum(time for _, time in self.parquet_download_times)
        avg_parquet_speed = total_parquet_bytes / total_parquet_time if total_parquet_time > 0 else 0

        total_time = time.time() - self.start_time

        return {
            "total_repos_processed": self.repos_processed,
            "total_files_processed": self.files_processed,
            "total_bytes_downloaded": self.bytes_downloaded,
            "avg_parquet_download_speed_mb_s": avg_parquet_speed / 1024 / 1024,
            "repos_per_second": self.repos_processed / total_time if total_time > 0 else 0,
            "files_per_second": self.files_processed / total_time if total_time > 0 else 0,
        }


@ray.remote
class ChunkWriter:
    def __init__(self, fs_protocol: str, fs_options: Optional[Dict], output_dir: str):
        if fs_protocol == "gcs":
            self.fs = gcsfs.GCSFileSystem(**fs_options)
        else:
            self.fs = fsspec.filesystem(fs_protocol, **(fs_options or {}))
        self.output_dir = output_dir

    def write_chunk(self, shard_id: int, chunk_id: int, records: List[Dict[str, Any]]):
        output_path = f"{self.output_dir}/shard_{shard_id:03d}_chunk_{chunk_id:05d}.jsonl.gz"
        print(f"Writing {output_path} with {len(records)} records")

        start_time = time.time()
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            for record in records:
                line = json.dumps(record) + "\n"
                gz.write(line.encode("utf-8"))

        with self.fs.open(output_path, "wb") as f:
            f.write(buffer.getvalue())

        duration = time.time() - start_time
        print(f"Wrote chunk {output_path} in {duration:.2f}s")

        return output_path


@ray.remote
class ChunkProcessor:
    def __init__(self):
        self.s3_for_downloads = s3fs.S3FileSystem()

    def process_chunk(
        self,
        shard_id: int,
        chunk_id: int,
        df: pd.DataFrame,
        writer: ray.actor.ActorHandle,
        stats: ray.actor.ActorHandle,
    ):
        chunk_start = time.time()
        current_chunk = []
        for _, repo in df.iterrows():
            repo_start = time.time()
            processed_files = []
            files_to_process = repo["files"]

            for file_info in files_to_process:
                blob_id = file_info["blob_id"]
                s3_path = f"softwareheritage/content/{blob_id}"

                try:
                    with self.s3_for_downloads.open(s3_path, "rb") as fin:
                        compressed_content = fin.read()
                        with gzip.GzipFile(fileobj=io.BytesIO(compressed_content)) as gz:
                            content = gz.read().decode("utf-8", errors="replace")

                    processed_files.append(
                        {
                            "blob_id": blob_id,
                            "length_bytes": file_info["length_bytes"],
                            "path": file_info["path"],
                            "content": content,
                        }
                    )
                except Exception as e:
                    print(f"Error processing blob {blob_id}: {e}")
                    continue

            repo_duration = time.time() - repo_start
            if processed_files:
                stats.add_repo_time.remote(len(processed_files), repo_duration)

                current_chunk.append({"repo_name": repo["repo_name"], "files": processed_files})
            if len(current_chunk) % 50 == 0:
                print(f"Processed {len(current_chunk)} repos in chunk {shard_id}:{chunk_id}")

        chunk_duration = time.time() - chunk_start
        print(f"Processed chunk {shard_id}:{chunk_id} in {chunk_duration:.2f}s")
        print("ChunkProcessor writing chunk")
        out_file = ray.get(writer.write_chunk.remote(shard_id, chunk_id, current_chunk))
        stats_data = ray.get(stats.get_stats.remote())
        print(
            f"Progress: {stats_data['total_repos_processed']} repos, "
            f"{stats_data['total_files_processed']} files, "
            f"speed: {stats_data['repos_per_second']:.2f} repos/s, "
            f"{stats_data['files_per_second']:.2f} files/s"
        )
        return out_file


@ray.remote
def process_parquet_file(
    repo_id: str,
    shard_id: int,
    writer: ray.actor.ActorHandle,
    stats: ray.actor.ActorHandle,
    processors: ray.util.ActorPool,
    chunk_size: int = 1000,
    existing_chunks: Optional[set[tuple[int, int]]] = None,
):
    """Process a single parquet shard from the dataset."""

    # Download parquet file
    print(f"Starting download of shard {shard_id}")
    parquet_start = time.time()

    # Download the parquet file
    print("Downloading parquet file")
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"data/train-{shard_id:05d}-of-00064.parquet",
        repo_type="dataset",
    )
    print(f"Downloaded parquet file to {parquet_path}")

    # Get file size and read the parquet file
    file_size = Path(parquet_path).stat().st_size
    parquet_duration = time.time() - parquet_start
    print(
        f"Downloaded {file_size/1024/1024:.2f}MB parquet file in {parquet_duration:.2f}s "
        f"({file_size/parquet_duration/1024/1024:.2f} MB/s)"
    )

    stats.add_parquet_time.remote(file_size, parquet_duration)

    pqfile = pq.ParquetFile(parquet_path).iter_batches(batch_size=chunk_size)

    num_chunks = 0
    for chunk_id, df in enumerate(pqfile):
        if existing_chunks and (shard_id, chunk_id) in existing_chunks:
            print(f"Skipping chunk {shard_id}:{chunk_id} as it already exists")
            continue

        processors.submit(lambda a, v: a.process_chunk.remote(*v), (shard_id, chunk_id, df.to_pandas(), writer, stats))
        num_chunks += 1

    chunks_written = []
    while processors.has_next():
        chunks_written.append(processors.get_next())

    print("Finished parquet file with chunks written:", chunks_written)
    return chunks_written


def process_dataset(
    repo_id: str,
    output_dir: str,
    num_workers: int = 8,
    fs_protocol: str = "file",
    fs_options: Optional[Dict] = None,
    start_shard: int = 0,
    end_shard: int = 64,
    chunk_size: int = 1000,
):
    """Process all parquet files from a HuggingFace dataset using Ray."""
    ray.init(num_cpus=num_workers)

    try:
        # Initialize actors
        writer = ChunkWriter.remote(fs_protocol, fs_options, output_dir)  # type: ignore
        stats = StatsActor.remote()  # type: ignore
        processors = ray.util.ActorPool([ChunkProcessor.remote() for _ in range(num_workers)])  # type: ignore

        # Initialize filesystem for output checking
        if fs_protocol == "gcs":
            fs = gcsfs.GCSFileSystem(**(fs_options or {}))
        else:
            fs = fsspec.filesystem(fs_protocol, **(fs_options or {}))

        # Check for existing chunks
        existing_chunks = set()
        if fs.exists(output_dir):
            for path in fs.glob(f"{output_dir}/*.jsonl.gz"):
                # look like shard_000_chunk_00000.jsonl.gz
                shard_id = int(path.split("_")[1])
                chunk_id = int(path.split("_")[-1].split(".")[0])
                existing_chunks.add((shard_id, chunk_id))
            print(f"Found {len(existing_chunks)} existing chunks")
        else:
            try:
                fs.mkdir(output_dir, create_parents=True)
            except Exception as e:
                print("Error creating output directory:", e)

        # Submit tasks for each shard
        tasks = []
        for shard_id in range(start_shard, end_shard):
            task = process_parquet_file.remote(
                repo_id, shard_id, writer, stats, processors, chunk_size=chunk_size, existing_chunks=existing_chunks
            )
            tasks.append(task)

        # Wait for completion with progress tracking
        for task in tqdm(tasks, desc="Processing shards"):
            ray.get(task)

            # Log current stats
            current_stats = ray.get(stats.get_stats.remote())
            print(
                "Current stats:\n"
                f"- Total repos processed: {current_stats['total_repos_processed']}\n"
                f"- Total files processed: {current_stats['total_files_processed']}\n"
                f"- Average parquet download speed: {current_stats['avg_parquet_download_speed_mb_s']:.2f} MB/s\n"
                f"- Processing speed: {current_stats['repos_per_second']:.2f} repos/s, "
                f"{current_stats['files_per_second']:.2f} files/s"
            )

        print("Successfully wrote all chunks")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset processing script using Ray")
    parser.add_argument("--repo_id", type=str, help="HuggingFace dataset repository ID")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed files")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of Ray workers")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Num repos per chunk")
    parser.add_argument("--fs_protocol", type=str, default="file", help="Filesystem protocol (e.g., 'file', 'gcs')")
    parser.add_argument("--project", type=str, help="GCP project ID (for GCS)")
    parser.add_argument("--token", type=str, help="Path to GCP credentials JSON (for GCS)")
    parser.add_argument("--start_shard", type=int, default=0, help="Starting shard ID")
    parser.add_argument("--end_shard", type=int, default=64, help="Ending shard ID (exclusive)")
    args = parser.parse_args()

    fs_options = None
    if args.fs_protocol == "gcs":
        fs_options = {"project": args.project, "token": args.token}

    process_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        fs_protocol=args.fs_protocol,
        fs_options=fs_options,
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        chunk_size=args.chunk_size,
    )

# python build_fim_dataset.py --repo_id bigcode/the-stack-v2-train-smol-ids --output_dir ./test-out --num_workers 4 --fs_protocol file
