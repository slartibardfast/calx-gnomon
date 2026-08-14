//! Shared fixture: a tiny two-shard HF checkpoint exercising every band,
//! a passthrough dtype, a 1-D fallback tensor, an index, and a config.

use std::fs;
use std::path::{Path, PathBuf};

pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let top = (b >> 16) as u16;
    let rest = b & 0xffff;
    let mut h = top;
    if rest > 0x8000 || (rest == 0x8000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

pub struct T<'a> {
    pub name: &'a str,
    pub dtype: &'a str,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn bf16_tensor<'a>(name: &'a str, shape: Vec<usize>, vals: &[f32]) -> T<'a> {
    let mut data = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        data.extend_from_slice(&f32_to_bf16_bits(*v).to_le_bytes());
    }
    T { name, dtype: "BF16", shape, data }
}

pub fn raw_tensor<'a>(name: &'a str, dtype: &'a str, shape: Vec<usize>, data: Vec<u8>) -> T<'a> {
    T { name, dtype, shape, data }
}

pub fn write_shard(path: &Path, tensors: &[T]) {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, t) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let end = offset + t.data.len();
        header.push_str(&format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            t.name,
            t.dtype,
            t.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
            offset,
            end
        ));
        offset = end;
    }
    header.push('}');
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for t in tensors {
        file.write_all(&t.data).unwrap();
    }
}

use std::io::Write;

/// Values chosen to land in every band: A (1.0, 0.5, and 1e-8 the
/// subnormal path), B (40000), C (1e30).
pub fn build_fixture(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let a_vals: Vec<f32> = vec![1.0, 0.5, -2.0, 1e-8, 4.0, -0.25];
    let b_vals: Vec<f32> = vec![40000.0, -1.0, 30000.0, 2.0, -3.0, 7.0, 0.5, 1.0];
    let one_d: Vec<f32> = vec![1.0, 40000.0, 1e30, 0.5];
    write_shard(
        &dir.join("model-00001-of-00002.safetensors"),
        &[
            bf16_tensor("model.w.a", vec![2, 3], &a_vals),
            bf16_tensor("model.w.big", vec![2, 3], &[1e30, 1.0, -1.0, 0.5, 2.0, -4.0]),
            raw_tensor("model.w.f32", "F32", vec![2, 2], vec![0u8; 16]),
        ],
    );
    write_shard(
        &dir.join("model-00002-of-00002.safetensors"),
        &[
            bf16_tensor("model.w.b", vec![2, 4], &b_vals),
            bf16_tensor("model.w.one_d", vec![4], &one_d),
            raw_tensor("model.w.i32", "I32", vec![2], vec![1u8, 0, 0, 0, 2, 0, 0, 0]),
        ],
    );
    let index = serde_json::json!({
        "metadata": {"total_size": 1},
        "weight_map": {
            "model.w.a": "model-00001-of-00002.safetensors",
            "model.w.big": "model-00001-of-00002.safetensors",
            "model.w.f32": "model-00001-of-00002.safetensors",
            "model.w.b": "model-00002-of-00002.safetensors",
            "model.w.one_d": "model-00002-of-00002.safetensors",
            "model.w.i32": "model-00002-of-00002.safetensors",
        }
    });
    fs::write(dir.join("model.safetensors.index.json"), serde_json::to_string(&index).unwrap()).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{"architectures":["T"],"torch_dtype":"bfloat16","vocab_size":8}"#,
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"a":1}"#).unwrap();
}

pub fn temp_fixture(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("calx-gnomon-test-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    build_fixture(&dir);
    dir
}

pub fn run_cli(args: &[&str]) -> (i32, String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_calx-gnomon"))
        .args(args)
        .output()
        .expect("run calx-gnomon");
    (
        out.status.code().unwrap() as i32,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Parse a written shard: (header json, tensor name -> (dtype, data bytes)).
pub fn read_shard(path: &Path) -> (serde_json::Value, Vec<(String, String, Vec<u8>)>) {
    let bytes = fs::read(path).unwrap();
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
    let base = 8 + hlen;
    let mut out = Vec::new();
    for (name, v) in hdr.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap().to_string();
        let (s, e) = (
            v["data_offsets"][0].as_u64().unwrap() as usize + base,
            v["data_offsets"][1].as_u64().unwrap() as usize + base,
        );
        out.push((name.clone(), dtype, bytes[s..e].to_vec()));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    (hdr, out)
}

pub fn tsv_rows(text: &str) -> Vec<Vec<String>> {
    text.trim_end().lines().map(|l| l.split('\t').map(String::from).collect()).collect()
}
