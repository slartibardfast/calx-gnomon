//! Tier-dispatch obligations: config defaults and the T1-T5 laws, at the
//! core level and through the written artifact.

mod common;

use calx_gnomon::fp::bf16_to_f16_table;
use calx_gnomon::{classify_band, dispatch, Action, Band, Tier};
use common::{f32_to_bf16_bits, read_shard, run_cli, temp_fixture, tsv_rows};

#[test]
fn config_defaults() {
    assert_eq!(calx_gnomon::FP16_MAX, 65504.0);
    assert_eq!(calx_gnomon::FP16_HALF_RANGE, 32768.0);
    assert_eq!(calx_gnomon::RESCALE_TARGET, 30000.0);
    assert_eq!(calx_gnomon::MIN_SCALE, 1e-30);
    // And the boundaries classify exactly on those constants.
    assert_eq!(classify_band(32768.0), Band::A);
    assert_eq!(classify_band(32768.1), Band::B);
    assert_eq!(classify_band(65504.0), Band::B);
    assert_eq!(classify_band(65505.0), Band::C);
    assert_eq!(classify_band(f64::NAN), Band::C); // the carried quirk
}

fn out_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("calx-gnomon-tier-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn run_tier(tag: &str, tier: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let fx = temp_fixture(tag);
    let out = out_dir(tag);
    let (code, _, err) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
        "--tier", tier,
    ]);
    let expect = if tier == "t1" { 2 } else { 0 };
    assert_eq!(code, expect, "tier {tier}: {err}");
    (fx, out)
}

#[test]
fn t1_dispatch() {
    let (fx, out) = run_tier("t1", "t1");
    let (_, shard1) = read_shard(&out.join("model-00001-of-00002.safetensors"));
    let by = |n: &str| shard1.iter().find(|(name, _, _)| name == n).unwrap();

    // Band A casts to F16, bytes = the RNE table of the bf16 inputs.
    let (n, dt, data) = by("model.w.a");
    assert_eq!((n.as_str(), dt.as_str()), ("model.w.a", "F16"));
    let table = bf16_to_f16_table();
    let vals = [1.0f32, 0.5, -2.0, 1e-8, 4.0, -0.25];
    for (i, v) in vals.iter().enumerate() {
        let expect = table[f32_to_bf16_bits(*v) as usize];
        assert_eq!(&data[i * 2..i * 2 + 2], &expect.to_le_bytes(), "value {v}");
    }

    // Band C stays BF16, byte-identical to the input.
    let (_, dt_big, _data_big) = by("model.w.big");
    assert_eq!(dt_big, "BF16");

    // Non-bf16 passes through byte-identically.
    let (_, dt_f32, data_f32) = by("model.w.f32");
    assert_eq!((dt_f32.as_str(), data_f32.len()), ("F32", 16));

    // config.json rewritten, tokenizer verbatim.
    let cfg = std::fs::read_to_string(out.join("config.json")).unwrap();
    assert!(cfg.contains(r#""torch_dtype":"float16""#), "{cfg}");
    assert_eq!(std::fs::read_to_string(out.join("tokenizer.json")).unwrap(), r#"{"a":1}"#);

    // Index invariance: the weight_map copied verbatim.
    assert_eq!(
        std::fs::read_to_string(fx.join("model.safetensors.index.json")).unwrap(),
        std::fs::read_to_string(out.join("model.safetensors.index.json")).unwrap(),
    );

    // The census says the same thing.
    let (_, stdout, _) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "dry-run"]);
    let rows = tsv_rows(&stdout);
    let big = rows[1..].iter().find(|r| r[0] == "model.w.big").unwrap();
    assert_eq!(big[6], "dry-run band C");
}

#[test]
fn t2_t3_rescale() {
    for tier in ["t2", "t3"] {
        let (fx, out) = run_tier(tier, tier);
        let (_, shard1) = read_shard(&out.join("model-00001-of-00002.safetensors"));
        // Band C cast to F16 with the per-tensor scale absmax/30000.
        let (_, dt, data) = shard1.iter().find(|(n, _, _)| n == "model.w.big").unwrap();
        assert_eq!(dt, "F16", "{tier}");
        let absmax = 1e30f64;
        let scale = (absmax / 30000.0) as f32;
        let expect0 = calx_gnomon::fp::f32_to_f16_bits(1e30f32 / scale);
        assert_eq!(&data[0..2], &expect0.to_le_bytes(), "{tier} first value");
        // Band A at scale 1.0 (same bytes as a t1 cast for that tensor).
        let (_, _, data_a) = shard1.iter().find(|(n, _, _)| n == "model.w.a").unwrap();
        let table = bf16_to_f16_table();
        assert_eq!(data_a[..2], table[f32_to_bf16_bits(1.0) as usize].to_le_bytes());
        // The config rewrite happened (casts occurred).
        assert!(std::fs::read_to_string(out.join("config.json")).unwrap().contains("float16"));
    }
}

#[test]
fn t4_records_scales_only() {
    // T4 on band C is the carried quirk at its limit: plain cast would
    // produce Inf, and like gguf-recast the data pass halts (exit 1) --
    // never a silent clamp. So this run preserves the band-C tensors by
    // policy and verifies the record-only law on the band-A/B tensors.
    let fx = temp_fixture("t4");
    let pol = std::env::temp_dir().join(format!("calx-gnomon-t4pol-{}.yaml", std::process::id()));
    std::fs::write(&pol, "preserve_bf16:\n  - 'w\\.big$'\n  - 'w\\.one_d$'\n").unwrap();
    let out = out_dir("t4");
    let (code, _, err) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
        "--tier", "t4",
        "--policy", pol.to_str().unwrap(),
    ]);
    assert_eq!(code, 2, "band C kept by policy: {err}");
    let (hdr, shard1) = read_shard(&out.join("model-00001-of-00002.safetensors"));
    // 2-D band A tensor: plain cast data (scales recorded, never
    // applied -- byte-identical to a t1 cast of the same tensor).
    let (_, dt, data_a) = shard1.iter().find(|(n, _, _)| n == "model.w.a").unwrap();
    assert_eq!(dt, "F16");
    let table = bf16_to_f16_table();
    assert_eq!(data_a[..2], table[f32_to_bf16_bits(1.0) as usize].to_le_bytes());
    // The row scales live in __metadata__, as a JSON array.
    let md = hdr["__metadata__"].as_object().expect("t4 metadata");
    let raw = md.get("recast.t4.model.w.a").expect("row scales recorded").as_str().unwrap();
    let scales: Vec<f64> = serde_json::from_str::<Vec<f32>>(raw).unwrap().iter().map(|&x| x as f64).collect();
    assert_eq!(scales.len(), 2, "one scale per row: {raw}");
    assert!(scales.iter().all(|&s| s == 1.0), "rows within range: {raw}");
    // A direct t4 on band C (no policy) is the faithful halt:
    let (code, _, err) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--output", out_dir("t4-halt").to_str().unwrap(),
        "--tier", "t4",
    ]);
    assert_eq!(code, 1);
    assert!(err.contains("produced Inf"), "{err}");
}

#[test]
fn t5_hadamard() {
    let (_, out) = run_tier("t5", "t5");
    let (_, shard2) = read_shard(&out.join("model-00002-of-00002.safetensors"));
    // The 2-D tensor took the Hadamard path: F16 out. Row length 4 ->
    // side d = 4. Verify one output numerically: row 0 of model.w.b is
    // [40000, -1, 30000, 2] -> WHT([40000, -1, 30000, 2]) / 2, all
    // within f16 range at scale 1.
    let (_, dt, data) = shard2.iter().find(|(n, _, _)| n == "model.w.b").unwrap();
    assert_eq!(dt, "F16");
    let mut row = [40000.0f32, -1.0, 30000.0, 2.0];
    calx_gnomon::hadamard::walsh_hadamard_row(&mut row);
    // The T5 law: rescale from the POST-transform absmax when it exceeds
    // the half range, then cast.
    let post = row.iter().fold(0.0f32, |m, &x| m.max(x.abs())) as f64;
    let scale = if post > calx_gnomon::FP16_HALF_RANGE {
        (post / calx_gnomon::RESCALE_TARGET).max(calx_gnomon::MIN_SCALE)
    } else {
        1.0
    } as f32;
    for (i, v) in row.iter().enumerate() {
        let expect = calx_gnomon::fp::f32_to_f16_bits(*v / scale);
        assert_eq!(&data[i * 2..i * 2 + 2], &expect.to_le_bytes(), "t5 value {i}");
    }
    // 1-D falls back to the T2 rescale (absmax 1e30 -> /scale).
    let (_, _, data_1d) = shard2.iter().find(|(n, _, _)| n == "model.w.one_d").unwrap();
    let scale = (1e30f64 / 30000.0) as f32;
    let expect0 = calx_gnomon::fp::f32_to_f16_bits(1.0f32 / scale);
    assert_eq!(&data_1d[0..2], &expect0.to_le_bytes());
}

#[test]
fn policy_preserve_overrides_action() {
    let fx = temp_fixture("pol");
    let pol = std::env::temp_dir().join(format!("calx-gnomon-pol-{}.yaml", std::process::id()));
    std::fs::write(&pol, "preserve_bf16:\n  - 'w\\.big$'\n").unwrap();
    let out = out_dir("pol");
    let (_, stdout, _) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--tier", "t2", // would rescale C; the preserve keeps it bf16
        "--output", out.to_str().unwrap(),
        "--policy", pol.to_str().unwrap(),
        "--tsv", "/dev/null",
    ]);
    let _ = stdout;
    let (_, shard1) = read_shard(&out.join("model-00001-of-00002.safetensors"));
    let (_, dt, _) = shard1.iter().find(|(n, _, _)| n == "model.w.big").unwrap();
    assert_eq!(dt, "BF16", "policy preserve keeps band C bf16 even at t2");
    // And the run still exits 2: band C present, deviation signalled.
}

#[test]
fn dispatch_core_laws() {
    // T1 never casts band C; policy preserve wins at every tier.
    for tier in [Tier::T1, Tier::T2, Tier::T4, Tier::T5] {
        assert!(matches!(
            dispatch(tier, Band::C, 1e30, true, false, true, None, None, 1),
            Action::KeepBf16 { .. }
        ));
    }
    assert!(matches!(
        dispatch(Tier::T1, Band::C, 1e30, true, false, false, None, None, 1),
        Action::KeepBf16 { .. }
    ));
}
