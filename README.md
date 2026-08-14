# calx-gnomon

The absmax-aware BF16→FP16 selective recast as one Rust tool over both artifact
classes — HF safetensors checkpoints and GGUF — all tiers. The gnomon is the
pointer that indicates: per tensor, which band it falls in (A, B, C by absmax)
and therefore what may be done to it.

Core/adapter seam: a Kani-proven artifact-generic core (band classification,
RNE cast, tier dispatch, census) plus `hf` and `gguf` adapters that move bytes
and never do arithmetic. Behaviour is specified in Allium; equivalence to the
incumbent `gguf-recast` is a byte-identical differential gate.

Home of record: the yarn-agentic host (`plan/0146-calx-gnomon`,
`call/0039`). Released into the public domain (Unlicense).
