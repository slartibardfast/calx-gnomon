//! The GGUF-tier acceptance gates (call/0039 P3, plan/0146):
//! byte-identity of the Q4_0 trunk repack against the oracle
//! (gguf-recast @ 5887dfa, itself byte-identical to the Python recipe),
//! plus the FP16 passthrough and mtp-splice behaviour that the plan/0147
//! flow carries and the oracle never had.
//!
//! Three hermetic tests run in CI against committed fixtures:
//! - trunk_byte_identical_to_oracle: fixtures/ar/reference-trunk.gguf is
//!   gguf-recast's own remix output over fixtures/ar/ckpt.
//! - f16_keeper_export_repacks: the same quants, keepers as FP16 (the
//!   shape an FP16-tuned AutoRound export carries).
//! - splice_path: the mtp splice from fixtures/ar/splice.
//!
//! The big gate is opt-in (multi-gigabyte host artifacts):
//!
//! ```sh
//! CALX_GNOMON_AR_SRC=/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/snapshots/abc86de19eb1ebbf6a7df4582341325c22ddcb7d \
//! CALX_GNOMON_AR_REF=/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf \
//! cargo test --release -- --ignored real_ar_byte_identical
//! ```
//!
//! The reference is sha256 656c898770e835d85ef0f4eee57b057f03d6d7e1d3e0a1be64aca1e0d7480422
//! (19,246,993,696 bytes, 866 tensors, 42 KVs), the Python-recipe GGUF the
//! qwen36 window shipped.

use std::io::Read;
use std::path::{Path, PathBuf};

use calx_gnomon::gguf::format::{Reader, Value};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name)
}

/// Stage a fixture checkpoint under a fixed directory name: the general.name
/// metadata falls back to model-id components, which must not vary with the
/// tmp dir's name.
fn stage(src: &Path, work: &Path) -> PathBuf {
    let dst = work.join("ar-fixture");
    copy_tree(src, &dst);
    dst
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let t = dst.join(e.file_name());
        std::fs::copy(e.path(), t).unwrap();
    }
}

fn bytes_identical(a: &Path, b: &Path) -> (bool, u64) {
    let mut fa = std::fs::File::open(a).unwrap();
    let mut fb = std::fs::File::open(b).unwrap();
    assert_eq!(
        fa.metadata().unwrap().len(),
        fb.metadata().unwrap().len(),
        "size mismatch"
    );
    let mut ba = vec![0u8; 4 << 20];
    let mut bb = vec![0u8; 4 << 20];
    let mut off: u64 = 0;
    loop {
        let na = fa.read(&mut ba).unwrap();
        let nb = fb.read(&mut bb).unwrap();
        assert_eq!(na, nb);
        if na == 0 {
            return (true, off);
        }
        if ba[..na] != bb[..nb] {
            return (false, off);
        }
        off += na as u64;
    }
}

#[test]
fn trunk_byte_identical_to_oracle() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-trunk");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = stage(&fixture("ar/ckpt"), &work);
    let out = work.join("out.gguf");

    calx_gnomon::gguf::run(&src, &out).expect("repack failed");
    let reference = fixture("ar/reference-trunk.gguf");
    let (ok, off) = bytes_identical(&out, &reference);
    assert!(ok, "first differing block at offset {off} (vs gguf-recast @ 5887dfa)");
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn f16_keeper_export_repacks() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-f16");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = stage(&fixture("ar/ckpt-f16"), &work);
    let out = work.join("out.gguf");

    calx_gnomon::gguf::run(&src, &out).expect("f16-keeper repack failed");

    // Same plan as the bf16 trunk: names, dtypes and shapes agree, only
    // the keeper data bytes differ (f16 vs bf16-derived f16 casts).
    let ours = Reader::open(&out).unwrap();
    let reference = Reader::open(&fixture("ar/reference-trunk.gguf")).unwrap();
    assert_eq!(ours.tensors.len(), reference.tensors.len());
    for (a, b) in ours.tensors.iter().zip(reference.tensors.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.dtype, b.dtype, "dtype of {}", a.name);
        assert_eq!(a.dims, b.dims, "shape of {}", a.name);
    }
    assert_eq!(ours.kvs.len(), reference.kvs.len());
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn splice_path() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-splice");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = stage(&fixture("ar/ckpt"), &work);
    let splice = fixture("ar/splice");
    let out = work.join("out.gguf");

    calx_gnomon::gguf::run_with_splice(&src, &splice, &out).expect("splice failed");

    let r = Reader::open(&out).unwrap();
    let reference = Reader::open(&fixture("ar/reference-trunk.gguf")).unwrap();
    // Fifteen mtp layers.* / fc / norms tensors plus one mtp.fc.bias.
    assert_eq!(r.tensors.len(), reference.tensors.len() + 16);

    let by_name = |n: &str| r.tensors.iter().find(|t| t.name == n).expect(n);
    let f16 = 1u32;
    let f32 = 0u32;

    // ggml naming and dtype routing of the spliced block.
    assert_eq!(by_name("blk.2.attn_q.weight").dtype, f16);
    assert_eq!(by_name("blk.2.attn_v.weight").dtype, f16);
    assert_eq!(by_name("blk.2.attn_output.weight").dtype, f16);
    assert_eq!(by_name("blk.2.ffn_gate.weight").dtype, f16);
    assert_eq!(by_name("blk.2.attn_norm.weight").dtype, f32);
    assert_eq!(by_name("blk.2.post_attention_norm.weight").dtype, f32);
    assert_eq!(by_name("blk.2.attn_q_norm.weight").dtype, f32);
    assert_eq!(by_name("blk.2.nextn.eh_proj.weight").dtype, f16);
    assert_eq!(by_name("blk.2.nextn.eh_proj.bias").dtype, f32);
    assert_eq!(by_name("blk.2.nextn.enorm.weight").dtype, f32);
    assert_eq!(by_name("blk.2.nextn.hnorm.weight").dtype, f32);
    assert_eq!(by_name("blk.2.nextn.shared_head_norm.weight").dtype, f32);

    // Values: the 2-D projections are byte-verbatim copies of the splice
    // source's f16; +1 norms are f16 + 1.0 in f32.
    let src_shard = splice.join("model-00001-of-00001.safetensors");
    let src_index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(splice.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let read_src = |name: &str| -> Vec<u16> {
        let meta = &src_index["weight_map"];
        assert!(meta[name].is_string());
        let raw = std::fs::read(&src_shard).unwrap();
        // Re-open via the crate's reader-free path: parse the header once.
        let hdr_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&raw[8..8 + hdr_len]).unwrap();
        let offs = hdr[name]["data_offsets"].as_array().unwrap();
        let s = offs[0].as_u64().unwrap() as usize + 8 + hdr_len;
        let e = offs[1].as_u64().unwrap() as usize + 8 + hdr_len;
        raw[s..e]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };

    let eh = r.tensor_data(&out, by_name("blk.2.nextn.eh_proj.weight")).unwrap();
    let src_eh = read_src("mtp.fc.weight");
    assert_eq!(eh.len(), src_eh.len() * 2);
    for (i, &h) in src_eh.iter().enumerate() {
        let got = u16::from_le_bytes([eh[2 * i], eh[2 * i + 1]]);
        assert_eq!(got, h, "eh_proj element {i} not byte-verbatim");
    }

    let enorm = r.tensor_data(&out, by_name("blk.2.nextn.enorm.weight")).unwrap();
    let src_enorm = read_src("mtp.pre_fc_norm_embedding.weight");
    assert_eq!(enorm.len(), src_enorm.len() * 4);
    for (i, &h) in src_enorm.iter().enumerate() {
        let want = f32::from_bits(calx_gnomon::fp::f16_to_f32_bits(h)) + 1.0f32;
        let got = f32::from_le_bytes(enorm[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(got.to_bits(), want.to_bits(), "enorm element {i}");
    }
    std::fs::remove_dir_all(&work).ok();
}

#[test]
#[ignore = "needs the AutoRound snapshot and the 19G Python reference; see module docs"]
fn real_ar_byte_identical() {
    let src = std::env::var("CALX_GNOMON_AR_SRC").expect("set CALX_GNOMON_AR_SRC");
    let reference = std::env::var("CALX_GNOMON_AR_REF").expect("set CALX_GNOMON_AR_REF");
    let out = std::env::temp_dir().join("calx-gnomon-real-ar-differential.gguf");

    calx_gnomon::gguf::run(Path::new(&src), &out).expect("repack failed");

    let (ok, off) = bytes_identical(&out, Path::new(&reference));
    assert!(ok, "first differing block at offset {off}");
    std::fs::remove_file(&out).ok();
}

/// The artifact must not carry any V-reorder metadata KV: the 3.8 lineage
/// does not bear it, and neither does the oracle's KV set.
#[test]
fn no_v_reorder_metadata_keys() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-kvs");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = stage(&fixture("ar/ckpt"), &work);
    let out = work.join("out.gguf");
    calx_gnomon::gguf::run(&src, &out).expect("repack failed");
    let r = Reader::open(&out).unwrap();
    for (k, _) in &r.kvs {
        assert!(
            !k.to_lowercase().contains("reorder"),
            "reorder metadata KV {k} must not be emitted"
        );
    }
    assert_eq!(
        r.get("general.architecture"),
        Some(&Value::Str("qwen35".into()))
    );
    std::fs::remove_dir_all(&work).ok();
}

/// The rejection guards bind the halt-class rules: a non-auto-round
/// config, an export that already carries mtp tensors (the splice is only
/// for exports whose module tree lacks them), and a splice directory with
/// no mtp tensors are all halt-class.
#[test]
fn repack_rejects_non_autoround() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-reject");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = stage(&fixture("ar/ckpt"), &work);
    // Rewrite the quant method: the checkpoint is no longer auto-round.
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(src.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut cfg = cfg;
    cfg["quantization_config"]["quant_method"] = serde_json::json!("gptq");
    std::fs::write(src.join("config.json"), cfg.to_string()).unwrap();
    let err = calx_gnomon::gguf::run(&src, &work.join("out.gguf"))
        .err()
        .expect("must reject a non-auto-round checkpoint");
    assert!(err.to_string().contains("auto-round"));
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn splice_rejects_export_with_mtp() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-reject2");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    // An export carrying mtp tensors cannot take the splice.
    let src = stage(&fixture("ar/splice"), &work);
    std::fs::copy(
        fixture("ar/ckpt/config.json"),
        src.join("config.json"),
    )
    .unwrap();
    let err = calx_gnomon::gguf::run_with_splice(
        &src,
        &fixture("ar/splice"),
        &work.join("out.gguf"),
    )
    .err()
    .expect("must reject an export that already carries mtp tensors");
    assert!(err.to_string().contains("mtp"));
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn splice_rejects_dir_without_mtp() {
    let work = std::env::temp_dir().join("calx-gnomon-parity-reject3");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    // ckpt-f16 carries no mtp.* tensor: unusable as a splice source.
    let src = stage(&fixture("ar/ckpt"), &work);
    let err = calx_gnomon::gguf::run_with_splice(
        &src,
        &fixture("ar/ckpt-f16"),
        &work.join("out.gguf"),
    )
    .err()
    .expect("must reject a splice dir without mtp tensors");
    assert!(err.to_string().contains("no mtp"));
    std::fs::remove_dir_all(&work).ok();
}
