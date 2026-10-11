"""Tokenize TinyStories (GPT-4, cleaned) at the character level.

`karpathy/tinystories-gpt4-clean` is one parquet file with a `text` column: 2,732,634
stories in 74 distinct ASCII characters (a-z, A-Z, 0-9, space, newline and
`!"$',-.:;?`). The vocabulary is these 74 characters (ids 1-74, in ASCII order) plus an
end-of-story token (id 0) after every story, i.e. 75 tokens. The splits are the ones
suggested by the dataset (rows 0-9,999 test, 10,000-19,999 val, the rest train).

Writes `train.bin`, `val.bin`, `test.bin` (uint8 token ids) and `meta.json` to `--out`.
Run with `python -I` (only pyarrow and numpy are needed).

Usage: python -I prepare_data.py --parquet <tinystories_gpt4_clean.parquet> --out <dir>
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

EOS = 0
SPLITS = {"test": (0, 10_000), "val": (10_000, 20_000), "train": (20_000, None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    texts = pq.read_table(args.parquet, columns=["text"]).column("text").to_pylist()
    print(f"{len(texts):,} stories", flush=True)
    chars = sorted(set().union(*(set(t) for t in texts)))
    assert len(chars) == 74 and all(ord(c) < 128 for c in chars), chars
    lut = np.zeros(128, dtype=np.uint8)
    lut[[ord(c) for c in chars]] = np.arange(1, len(chars) + 1, dtype=np.uint8)

    meta = {"vocab_size": len(chars) + 1, "eos": EOS, "chars": "".join(chars), "splits": {}}
    for name, (start, stop) in SPLITS.items():
        stop = len(texts) if stop is None else stop
        joined = "\0".join(texts[start:stop]) + "\0"  # "\0" marks the end of every story.
        raw = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
        tokens = np.where(raw == 0, EOS, lut[raw]).astype(np.uint8)
        assert np.all((tokens > 0) | (raw == 0))
        tokens.tofile(os.path.join(args.out, f"{name}.bin"))
        meta["splits"][name] = {"rows": [start, stop], "stories": stop - start, "tokens": int(tokens.size)}
        print(f"{name}: {stop - start:,} stories, {tokens.size:,} tokens", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
