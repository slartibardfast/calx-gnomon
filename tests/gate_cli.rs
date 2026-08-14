//! Gate-CLI obligations (specs/recast.obligations): invocation, lifecycle,
//! exit codes, census schema and completeness. Each test name matches a
//! `test:` disposition in the manifest.

mod common;

use common::{run_cli, temp_fixture, tsv_rows};

#[test]
fn gate_cli_invocation() {
    // Missing args: halt-class exit 1 with usage.
    let (code, _, err) = run_cli(&[]);
    assert_eq!(code, 1);
    assert!(err.contains("--input is required"), "{err}");
    // Unknown tier: halt-class.
    let fx = temp_fixture("inv");
    let (code, _, err) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "t9"]);
    assert_eq!(code, 1);
    assert!(err.contains("unknown tier"), "{err}");
    // Unknown flag: halt-class.
    let (code, _, _) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "dry-run", "--wat"]);
    assert_eq!(code, 1);
}

#[test]
fn run_lifecycle_completes() {
    // t2 rescales every band C: a completed, clean run exits 0.
    let fx = temp_fixture("complete");
    let out = std::env::temp_dir().join(format!("calx-gnomon-out-complete-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let (code, _, err) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
        "--tier", "t2",
    ]);
    assert_eq!(code, 0, "stderr: {err}");
    assert!(err.contains("tensors"), "{err}");
    assert!(out.join("config.json").is_file());
}

#[test]
fn run_lifecycle_halts() {
    // A corrupt shard (header length exceeds file): halt-class exit 1,
    // and no output artifact is trusted.
    let fx = temp_fixture("halt");
    let bad = fx.join("model-00001-of-00002.safetensors");
    std::fs::write(&bad, vec![0xffu8; 8]).unwrap();
    let out = std::env::temp_dir().join(format!("calx-gnomon-out-halt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let (code, _, err) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
        "--tier", "t1",
    ]);
    assert_eq!(code, 1);
    assert!(err.contains("header length"), "{err}");
    // A missing input dir halts too.
    let (code, _, _) = run_cli(&["--input", "/nonexistent-calx-gnomon", "--tier", "dry-run"]);
    assert_eq!(code, 1);
}

#[test]
fn exit_codes_distinct() {
    // 0 clean (t2), 2 completed-with-kept-band-C (t1), 1 halt (corrupt):
    // three classes, three codes, observed once each.
    let fx = temp_fixture("codes");
    let tsv_path = std::env::temp_dir().join(format!("calx-gnomon-codes-{}.tsv", std::process::id()));
    let out = std::env::temp_dir().join(format!("calx-gnomon-out-codes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let (c_dry, _, _) = run_cli(&[
        "--input", fx.to_str().unwrap(), "--tier", "dry-run", "--tsv", tsv_path.to_str().unwrap(),
    ]);
    let (c_t1, _, _) = run_cli(&[
        "--input", fx.to_str().unwrap(), "--tier", "t1", "--output", out.to_str().unwrap(),
    ]);
    assert_eq!((c_dry, c_t1), (0, 2));
}

#[test]
fn dry_run_writes_nothing() {
    let fx = temp_fixture("dry");
    let before: Vec<_> = std::fs::read_dir(&fx)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let (code, stdout, _) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "dry-run"]);
    assert_eq!(code, 0);
    assert!(stdout.starts_with("name\t"), "census went to stdout");
    let after: Vec<_> = std::fs::read_dir(&fx)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(before.len(), after.len(), "dry-run touched the input dir");
    // And no output artifact exists: an acting tier without --output is
    // itself a halt-class usage error (exit 1), never a silent no-op.
    let (code, _, err) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "t1"]);
    assert_eq!(code, 1);
    assert!(err.contains("--output is required"), "{err}");
}

#[test]
fn exit_code_signals_deviation() {
    // The fixture carries exactly one band-C tensor (model.w.big): t1
    // keeps it bf16, the run completes, exit 2. t2 rescales it: exit 0.
    let fx = temp_fixture("dev");
    let out = std::env::temp_dir().join(format!("calx-gnomon-out-dev-{}", std::process::id()));
    for tier in ["t1", "t2"] {
        let _ = std::fs::remove_dir_all(&out);
        let (code, _, err) = run_cli(&[
            "--input", fx.to_str().unwrap(),
            "--output", out.to_str().unwrap(),
            "--tier", tier,
        ]);
        match tier {
            "t1" => assert_eq!(code, 2, "stderr: {err}"),
            "t2" => assert_eq!(code, 0, "stderr: {err}"),
            _ => unreachable!(),
        }
    }
}

#[test]
fn census_tsv_schema() {
    // The StableCensusSchema law: gguf-recast's seven columns +
    // n_subnormal appended, exactly, in order.
    let fx = temp_fixture("tsv");
    let tsv = std::env::temp_dir().join(format!("calx-gnomon-tsv-{}.tsv", std::process::id()));
    let (code, stdout, _) = run_cli(&[
        "--input", fx.to_str().unwrap(),
        "--tier", "dry-run",
        "--tsv", tsv.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.is_empty(), "census goes to --tsv, not stdout, when given");
    let text = std::fs::read_to_string(&tsv).unwrap();
    let rows = tsv_rows(&text);
    assert_eq!(
        rows[0],
        vec!["name", "src_type", "out_type", "band", "absmax", "n_elem", "note", "n_subnormal"],
    );
    // A passthrough row carries "-" in the band column.
    let f32_row = rows.iter().find(|r| r[1] == "F32").expect("F32 passthrough row");
    assert_eq!(f32_row[3], "-");
    assert_eq!(f32_row[6], "passthrough");
    // Every row has exactly 8 columns.
    assert!(rows.iter().all(|r| r.len() == 8));
}

#[test]
fn census_classifies_every_tensor() {
    // Every tensor appears exactly once, with the band the absmax law
    // assigns: w.a -> A, w.b -> B, w.big -> C, 1-D included.
    let fx = temp_fixture("classify");
    let (_, stdout, _) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "dry-run"]);
    let rows = tsv_rows(&stdout);
    let names: Vec<&str> = rows[1..].iter().map(|r| r[0].as_str()).collect();
    for expect in ["model.w.a", "model.w.big", "model.w.f32", "model.w.b", "model.w.one_d", "model.w.i32"] {
        assert_eq!(names.iter().filter(|&&n| n == expect).count(), 1, "{expect} once in {names:?}");
    }
    let band_of = |name: &str| {
        rows[1..].iter().find(|r| r[0] == name).unwrap()[3].clone()
    };
    assert_eq!(band_of("model.w.a"), "A");
    assert_eq!(band_of("model.w.b"), "B");
    assert_eq!(band_of("model.w.big"), "C");
    assert_eq!(band_of("model.w.one_d"), "C"); // 1e30 present
}

#[test]
fn census_completeness() {
    // The index's weight_map names and the census rows are the same set.
    let fx = temp_fixture("census");
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let (_, stdout, _) = run_cli(&["--input", fx.to_str().unwrap(), "--tier", "dry-run"]);
    let rows = tsv_rows(&stdout);
    let mut names: Vec<&str> = rows[1..].iter().map(|r| r[0].as_str()).collect();
    names.sort_unstable();
    let mut want: Vec<&str> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    want.sort_unstable();
    assert_eq!(names, want);
}
