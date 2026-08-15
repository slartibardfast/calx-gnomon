#!/usr/bin/env python3
"""Deterministic synthetic AutoRound export fixture for the calx-gnomon
GGUF-tier parity gate (plan/0146 P3).

Emits a tiny Qwen3.5-shaped AR checkpoint (fixtures/ar/ckpt) exercising
every remix source class (Q4_0 trunk, V-row qkv, V-row z, V-col AR16,
all five bf16 conversion paths), and an FP16 splice directory
(fixtures/ar/splice) with mtp.* tensors for the splice-path tests.
Pure numpy; a seeded LCG keeps regeneration stable.
"""
import json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "fixtures" / "ar"

RS = np.random.RandomState(20260815)

def bf16_bits(arr: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit truncation, stored as uint16 (numpy has no bf16)."""
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

def st_bytes(tensors: dict, bf16_names: set) -> bytes:
    header = {}
    blobs = []
    off = 0
    for name in sorted(tensors):
        arr = tensors[name]
        if name in bf16_names:
            dtype, data = "BF16", bf16_bits(arr).tobytes()
        else:
            dtype = {"float16": "F16", "float32": "F32", "int32": "I32"}[str(arr.dtype)]
            data = arr.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [off, off + len(data)],
        }
        blobs.append(data)
        off += len(data)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(hdr)) + hdr + b"".join(blobs)

def quant(in_f: int, out_f: int, prefix: str, out: dict):
    q = RS.randint(0, 2**32, size=(in_f // 8, out_f)).astype(np.int32)
    sc = (RS.rand(in_f // 128, out_f).astype(np.float32) * 0.1).astype(np.float16)
    out[f"{prefix}.qweight"] = q
    out[f"{prefix}.scales"] = sc

CKPT, SPLICE = {}, {}

# Layer 0 (linear attention) quantized projections.
quant(256, 384, "model.layers.0.linear_attn.in_proj_qkv", CKPT)
quant(256, 256, "model.layers.0.linear_attn.in_proj_z", CKPT)
quant(256, 128, "model.layers.0.linear_attn.out_proj", CKPT)
# Layer 1 (full attention) + MLP trunk quants.
for p, (inf, outf) in {
    "model.layers.1.self_attn.q_proj": (128, 128),
    "model.layers.1.self_attn.k_proj": (128, 128),
    "model.layers.1.self_attn.v_proj": (128, 128),
    "model.layers.1.self_attn.o_proj": (128, 128),
    "model.layers.1.mlp.gate_proj": (128, 256),
    "model.layers.1.mlp.up_proj": (128, 256),
    "model.layers.1.mlp.down_proj": (256, 128),
}.items():
    quant(inf, outf, p, CKPT)

def bf16(shape, out=CKPT, name=None, scale=1.0):
    arr = (RS.randn(*shape) * scale).astype(np.float16).astype(np.float16)
    out[name] = arr
    return arr

CKPT["model.embed_tokens.weight"] = (RS.randn(64, 128) * 0.1).astype(np.float16)
CKPT["lm_head.weight"] = (RS.randn(64, 128) * 0.1).astype(np.float16)
CKPT["model.norm.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
bf16((128,), name="model.layers.0.input_layernorm.weight", scale=0.02)
bf16((128,), name="model.layers.0.post_attention_layernorm.weight", scale=0.02)
bf16((8, 128), name="model.layers.0.linear_attn.in_proj_a.weight", scale=0.01)
bf16((8, 128), name="model.layers.0.linear_attn.in_proj_b.weight", scale=0.01)
bf16((384, 1, 4), name="model.layers.0.linear_attn.conv1d.weight", scale=0.1)
bf16((8,), name="model.layers.0.linear_attn.A_log", scale=0.5)
bf16((8,), name="model.layers.0.linear_attn.dt_bias", scale=0.05)
bf16((256,), name="model.layers.0.linear_attn.norm.weight", scale=0.02)
bf16((16,), name="model.layers.1.self_attn.q_norm.weight", scale=0.02)
bf16((16,), name="model.layers.1.self_attn.k_norm.weight", scale=0.02)
bf16((128,), name="model.layers.1.input_layernorm.weight", scale=0.02)
bf16((128,), name="model.layers.1.post_attention_layernorm.weight", scale=0.02)

# ---- config / tokenizer / generation_config for the ckpt ----
config = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "_name_or_path": "qwen/ar-fixture",
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 16,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 1,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_theta": 5000000.0},
    "quantization_config": {
        "quant_method": "auto-round",
        "bits": 4,
        "sym": True,
        "group_size": 128,
    },
}
tok = {
    "version": "1.0",
    "added_tokens": [
        {"id": 0, "content": "", "special": True},
        {"id": 1, "content": "", "special": True},
    ],
    "model": {
        "type": "gpt2",
        "vocab": {f"t{i}": i + 2 for i in range(62)},
        "merges": [],
    },
}
tokcfg = {
    "eos_token": {"content": "", "special": True},
    "bos_token": {"content": "", "special": True},
    "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
}

SPLICE["model.layers.0.self_attn.q_proj.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.fc.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.fc.bias"] = (RS.randn(128) * 0.05).astype(np.float16)
SPLICE["mtp.norm.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
SPLICE["mtp.pre_fc_norm_embedding.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
SPLICE["mtp.pre_fc_norm_hidden.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
SPLICE["mtp.layers.0.input_layernorm.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
SPLICE["mtp.layers.0.post_attention_layernorm.weight"] = (RS.randn(128) * 0.02).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.q_proj.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.k_proj.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.v_proj.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.o_proj.weight"] = (RS.randn(128, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.q_norm.weight"] = (RS.randn(16) * 0.02).astype(np.float16)
SPLICE["mtp.layers.0.self_attn.k_norm.weight"] = (RS.randn(16) * 0.02).astype(np.float16)
SPLICE["mtp.layers.0.mlp.gate_proj.weight"] = (RS.randn(256, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.mlp.up_proj.weight"] = (RS.randn(256, 128) * 0.1).astype(np.float16)
SPLICE["mtp.layers.0.mlp.down_proj.weight"] = (RS.randn(128, 256) * 0.1).astype(np.float16)

KEEP_BF16 = {n for n, a in CKPT.items() if str(a.dtype) == "float16"}

def write_ckpt(dirpath: Path, tensors: dict, bf16_names: set):
    dirpath.mkdir(parents=True, exist_ok=True)
    part = "model-00001-of-00001.safetensors"
    blob = st_bytes(tensors, bf16_names)
    (dirpath / part).write_bytes(blob)
    idx = {"metadata": {"total_size": len(blob)},
           "weight_map": {n: part for n in sorted(tensors)}}
    (dirpath / "model.safetensors.index.json").write_text(
        json.dumps(idx, indent=2))

# ckpt: BF16 keepers, the oracle-parity shape (3.6 lineage).
write_ckpt(AR / "ckpt", CKPT, KEEP_BF16)
# ckpt-f16: identical bytes except the keepers land as FP16, the shape the
# plan/0147 quant export carries (FP16 tuning input).
write_ckpt(AR / "ckpt-f16", CKPT, set())
(AR / "ckpt" / "config.json").write_text(json.dumps(config, indent=2))
(AR / "ckpt" / "tokenizer.json").write_text(json.dumps(tok))
(AR / "ckpt" / "tokenizer_config.json").write_text(json.dumps(tokcfg))
(AR / "ckpt" / "generation_config.json").write_text(json.dumps(
    {"top_k": 20, "top_p": 0.95, "temperature": 1.0, "min_p": 0.001}))
for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
          "generation_config.json"):
    (AR / "ckpt-f16" / f).write_text((AR / "ckpt" / f).read_text())
write_ckpt(AR / "splice", SPLICE, set())
print(f"wrote {AR}/ckpt (BF16 keepers), {AR}/ckpt-f16 (FP16 keepers), "
      f"{AR}/splice ({len(SPLICE)} tensors)")
