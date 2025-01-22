import json
import sys

import numpy as np
from dask.array.slicing import shuffle_slice
from dask.bag import read_text
from dask.diagnostics import ProgressBar


def main():
    in_files = sys.argv[1]
    out_files = sys.argv[2]
    bag = read_text(in_files).map(json.loads)

    def flatten_files(x):
        repo_name = x["repo_name"]
        for file in x["files"]:
            content = file["content"]
            path = file["path"]
            blob_id = file["blob_id"]
            yield {"repo_name": repo_name, "path": path, "blob_id": blob_id, "content": content}

    with ProgressBar():
        df = (
            bag.map(flatten_files)
            .flatten()
            .to_dataframe(meta={"repo_name": str, "path": str, "blob_id": str, "content": str})
        )
        d_arr = df.to_dask_array(True)
        df_len = len(df)
        np.random.seed(42)
        index = np.random.choice(df_len, df_len, replace=False)
        d_arr = shuffle_slice(d_arr, index)
        df = d_arr.to_dask_dataframe(columns=df.columns)
        df.to_json(out_files, orient="records", lines=True, compression="gzip")


if __name__ == "__main__":
    main()
