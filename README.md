# calx-gnomon

The absmax-aware BF16→FP16 selective recast as one Rust tool over both artifact
classes — HF safetensors checkpoints and GGUF — all tiers. The gnomon is the
pointer that indicates: per tensor, which band it falls in (A, B, C by absmax)
and therefore what may be done to it.
Core/adapter seam: a Kani-proven artifact-generic core (band
classification, RNE cast, tier dispatch, census), the `hf` recast adapter,
and the `gguf` repack tier (AutoRound INT4 checkpoint -> Q4_0/Q4_0_AR16
GGUF with the optional mtp splice), all moving bytes and never doing
arithmetic the core has not decided. Behaviour is specified in Allium;
equivalence to the incumbent `gguf-recast` is a byte-identical differential
gate.

## Status

- **hf adapter**: read (single shard or index-directed shard set), census,
  dry-run, and tiers t1-t5 over HF safetensors checkpoints, writing a new
  checkpoint with the same tensor -> shard map, a surgical
  `torch_dtype` rewrite in config.json, and every other file verbatim.
- **gguf repack tier** (P3): AutoRound INT4 HF export -> GGUF (Q4_0 trunk,
  V-row-perm Q4_0, V-col-perm Q4_0_AR16 at type id 42), home port of
  gguf-recast's remix, byte-identical to it at its recorded pin (gated over
  the committed synthetic fixture and the real nineteen-gigabyte Qwen3.6
  export, opt-in). Adds two behaviours the incumbent lacks: FP16 keepers in
  the export (the FP16-tuning-input flow) and the mtp splice from a recast
  FP16 checkpoint (`--splice`) for exports whose module tree carries no mtp
  modules (Qwen3.8, plan/0147 D16). Recast-over-GGUF tiers (t1-t5 on GGUF
  artifacts) are still the incumbent's until the full retirement.

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
- **repack parity** (P3): `tests/repack_differential.rs`. Hermetic:
  `fixtures/ar/` (a synthetic AR export, BF16 and FP16 keeper variants, an
  FP16 splice dir, and the gguf-recast-emitted reference GGUF). Opt-in big
  gate: `CALX_GNOMON_AR_SRC` + `CALX_GNOMON_AR_REF` over the real Qwen3.6
  AutoRound snapshot and its Python-recipe reference (see the test's module
  docs).

## Build

Hermetic offline build (vendored dependencies):

```
cargo build --release --locked --offline
```

Usage:

```
calx-gnomon --input <hf-checkpoint-dir> --tier <dry-run|t1|t2|t3|t4|t5> \
    [--output <dir>] [--tsv <file>] [--policy <file>] [--force-rescale]

calx-gnomon repack --model-dir <auto-round-export-dir> --outfile <out.gguf> \
    [--splice <fp16-checkpoint-dir>]
```

Exit codes (the gate contract, `specs/recast.allium` surface GateCLI):
0 clean; 2 completed with kept band-C tensors (artifact valid, deviation
signalled); 1 halt-class failure.

Home of record: the yarn-agentic host (`plan/0146-calx-gnomon`,
`call/0039`). Released into the public domain (Unlicense).

