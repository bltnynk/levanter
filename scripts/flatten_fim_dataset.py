#!/usr/bin/env python3
import gzip
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Iterator, List

import fsspec


def batch_iterator(iterator: Iterator, batch_size: int) -> Iterator[List]:
    """Yield batches from an iterator."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def process_jsonl(input_path: str, output_path: str, batch_size: int = 1000) -> None:
    """
    Read a JSONL file and flatten it along the files axis with batched processing.

    Args:
        input_path: Path to input JSONL file (can be gzipped)
        output_path: Path to write flattened output (will be gzipped)
        batch_size: Number of lines to process in each batch
    """
    fs = fsspec.filesystem("file")

    def line_generator(file_obj):
        """Generate lines from file object."""
        for line in file_obj:
            yield line

    # Open output file with gzip compression
    with gzip.open(output_path, "wt", compresslevel=6) as out_f:
        # Open input file with inferred compression
        with fs.open(input_path, "r", compression="infer") as in_f:
            # Process in batches
            for batch in batch_iterator(line_generator(in_f), batch_size):
                flattened_records = []

                # Process each line in the batch
                for line in batch:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}", file=sys.stderr)
                        continue

                    repo_name = record.get("repo_name")
                    files = record.get("files", [])

                    # Create one record per file
                    for file in files:
                        flattened_records.append(
                            {"repo_name": repo_name, "files": [file]}  # Wrap single file in list to maintain schema
                        )

                # Write entire batch at once
                if flattened_records:
                    out_f.write("\n".join(json.dumps(record) for record in flattened_records) + "\n")


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python flatten_jsonl.py <input_path> <output_path> [batch_size]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    if not Path(input_path).exists():
        print(f"Input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    process_jsonl(input_path, output_path, batch_size)


if __name__ == "__main__":
    main()
