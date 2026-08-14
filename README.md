# calx-gnomon

The absmax-aware BF16→FP16 selective recast as one Rust tool over both artifact
classes — HF safetensors checkpoints and GGUF — all tiers. The gnomon is the
pointer that indicates: per tensor, which band it falls in (A, B, C by absmax)
and therefore what may be done to it.
Core/adapter seam: a Kani-proven artifact-generic core (band
classification, RNE cast, tier dispatch, census) plus `hf` and `gguf`
adapters that move bytes and never do arithmetic. Behaviour is specified
in Allium; equivalence to the incumbent `gguf-recast` is a byte-identical
differential gate.

## Status

- **hf adapter**: read (single shard or index-directed shard set), census,
  dry-run, and tiers t1-t5 over HF safetensors checkpoints, writing a new
  checkpoint with the same tensor -> shard map, a surgical
  `torch_dtype` rewrite in config.json, and every other file verbatim.
- **gguf adapter**: lands with the port (byte-parity gate against
  gguf-recast at its recorded pin); `--input *.gguf` is a halt-class error
  until then.

## Verification lanes

- **Allium**: `specs/recast.allium` (check + analyse clean; 43 obligations
  dispositioned in `specs/recast.obligations`).
- **Kani**: `cargo kani` proves the cast law (bf16 -> f16 equals the
  RNE composition, universal over the 16-bit domain), the subnormal
  accounting law, the band classification law, and band-C-never-cast.
- **cargo test**: the gate-CLI laws (exit codes 0/2/1, TSV schema, census
  completeness, dry-run writes nothing) and the tier dispatch (t1 keep/cast,
  t2/t3 rescale, t4 record-only, t5 Hadamard).
- **torch parity**: opt-in, `CALX_GNOMON_TORCH_FIXTURE=<file>` (65536
  bf16/f16 pairs from the reference venv); skips loudly without it.

## Build

Hermetic offline build (vendored dependencies):

```
cargo build --release --locked --offline
```

Exit codes (the gate contract, `specs/recast.allium` surface GateCLI):
0 clean; 2 completed with kept band-C tensors (artifact valid, deviation
signalled); 1 halt-class failure.

Home of record: the yarn-agentic host (`plan/0146-calx-gnomon`,
`call/0039`). Released into the public domain (Unlicense).

