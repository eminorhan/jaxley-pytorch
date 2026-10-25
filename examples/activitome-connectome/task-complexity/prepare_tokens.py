"""BPE tokenization of TinyStories (GPT-4, cleaned) for the `tokens` task.

Trains a byte-level BPE tokenizer (Hugging Face `tokenizers`; GPT-2 pre-tokenization;
`--vocab` tokens, among them the end-of-story token `<|endoftext|>` = id 0) on the
training split, then writes `train.bin`, `val.bin`, `test.bin` (uint16 token ids, an
end-of-story token after every story), `tokenizer.json` and `meta.json` to `--out`. The
splits are those of examples/language_modeling/prepare_data.py (rows 0-9,999 test,
10,000-19,999 val, the rest train), whose character-level files the `chars` task uses;
`meta.json` records the characters per token (a story's end counts as one character, as
in the character-level files).

Usage: python -I prepare_tokens.py --parquet <tinystories_gpt4_clean.parquet> --vocab 4096 --out <dir>
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"
SPLITS = {"test": (0, 10_000), "val": (10_000, 20_000), "train": (20_000, None)}
BATCH = 10_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    texts = pq.read_table(args.parquet, columns=["text"]).column("text").to_pylist()
    print(f"{len(texts):,} stories ({time.time() - t0:.0f} s)", flush=True)
    train = texts[SPLITS["train"][0]:]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=[EOS], show_progress=False)
    tok.train_from_iterator((train[i : i + BATCH] for i in range(0, len(train), BATCH)), trainer=trainer,
                            length=len(train))
    assert tok.token_to_id(EOS) == 0 and tok.get_vocab_size() == args.vocab, tok.get_vocab_size()
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"tokenizer trained ({time.time() - t0:.0f} s)", flush=True)

    encode = getattr(tok, "encode_batch_fast", tok.encode_batch)
    meta = {"vocab_size": args.vocab, "eos": 0, "dtype": "uint16", "tokenizer": "tokenizer.json",
            "pre_tokenizer": "ByteLevel(add_prefix_space=False)", "splits": {}}
    for name, (start, stop) in SPLITS.items():
        stop = len(texts) if stop is None else stop
        n_tokens, n_chars = 0, 0
        with open(os.path.join(args.out, f"{name}.bin"), "wb") as f:
            for i in range(start, stop, BATCH):
                batch = texts[i : min(i + BATCH, stop)]
                ids = []
                for enc in encode(batch):
                    ids.extend(enc.ids)
                    ids.append(0)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(f)
                n_tokens += arr.size
                n_chars += sum(len(t) + 1 for t in batch)
        meta["splits"][name] = {"rows": [start, stop], "stories": stop - start, "tokens": n_tokens,
                                "chars": n_chars, "chars_per_token": n_chars / n_tokens}
        print(f"{name}: {stop - start:,} stories, {n_tokens:,} tokens, {n_chars / n_tokens:.3f} characters "
              f"per token ({time.time() - t0:.0f} s)", flush=True)

    # The ids decode back to the stories.
    for text in texts[10_000:10_100]:
        assert tok.decode(tok.encode(text).ids) == text
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
