//! The HF-safetensors adapter: mmap-read a checkpoint (single shard or
//! index-directed shard set), census it, and (non-dry-run) write the
//! recast checkpoint with the same tensor -> shard map. Ported reader
//! semantics from gguf-recast's `safetensors.rs`; the writer is new (the
//! artifact class gguf-recast never wrote).

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;

use crate::fp::{bf16_to_f16_table, bf16_to_f32, f32_to_f16_bits, is_subnormal_path};
use crate::hadamard::{largest_pow2_le, walsh_hadamard_row};
use crate::{classify_band, dispatch, Action, Band, CensusRow, Tier};

/// One tensor as the safetensors header describes it.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub bpe: usize,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl TensorEntry {
    pub fn n_elem(&self) -> usize {
        self.shape.iter().product::<usize>().max((self.end - self.start) as usize / self.bpe.max(1))
    }
}

pub struct Shard {
    pub path: PathBuf,
    pub mmap: Mmap,
    /// Tensors in data order (ascending start offset).
    pub tensors: Vec<TensorEntry>,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    /// The tensor -> shard map (index.json's weight_map when present).
    pub weight_map: BTreeMap<String, usize>,
}

fn bpe_of(dtype: &str) -> usize {
    match dtype {
        "BF16" | "F16" | "I16" => 2,
        "F32" | "I32" => 4,
        "F64" | "I64" => 8,
        "U8" | "I8" | "BOOL" => 1,
        _ => 0, // unknown: pass through bytes, n_elem from shape
    }
}

fn parse_shard_header(bytes: &[u8], path: &Path) -> io::Result<(u64, Vec<TensorEntry>)> {
    if bytes.len() < 8 {
        return Err(bad(path, "safetensors file shorter than the header length prefix"));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if hlen > bytes.len() - 8 {
        return Err(bad(path, "safetensors header length exceeds file size"));
    }
    let hdr: Value = serde_json::from_slice(&bytes[8..8 + hlen])
        .map_err(|e| bad(path, &format!("safetensors header JSON: {e}")))?;
    let obj = hdr.as_object().ok_or_else(|| bad(path, "safetensors header is not an object"))?;
    let mut entries = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(path, &format!("tensor {name}: no dtype")))?;
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
            .ok_or_else(|| bad(path, &format!("tensor {name}: no shape")))?;
        let offs = v
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| bad(path, &format!("tensor {name}: no data_offsets")))?;
        let (Some(s), Some(e)) = (offs.first().and_then(Value::as_u64), offs.get(1).and_then(Value::as_u64))
            else {
                return Err(bad(path, &format!("tensor {name}: bad data_offsets")));
            };
        entries.push(TensorEntry {
            name: name.clone(),
            dtype: dtype.to_string(),
            bpe: bpe_of(dtype),
            shape,
            start: s,
            end: e,
        });
    }
    let data_base = 8 + hlen;
    for t in &mut entries {
        t.start += data_base as u64;
        t.end += data_base as u64;
    }
    entries.sort_by_key(|t| t.start);
    Ok((data_base as u64, entries))
}

fn bad(path: &Path, msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {msg}", path.display()))
}

impl Checkpoint {
    /// Open a checkpoint directory: index-directed when
    /// `model.safetensors.index.json` is present, else every
    /// `*.safetensors` in sorted filename order.
    pub fn open(dir: &Path) -> io::Result<Checkpoint> {
        if !dir.is_dir() {
            return Err(bad(dir, "not a directory (the hf adapter takes a checkpoint dir)"));
        }
        let index = dir.join("model.safetensors.index.json");
        let (shard_names, weight_map): (Vec<String>, BTreeMap<String, usize>) =
            if index.is_file() {
                let v: Value = serde_json::from_str(&fs::read_to_string(&index)?)
                    .map_err(|e| bad(&index, &format!("index JSON: {e}")))?;
                let wm = v
                    .get("weight_map")
                    .and_then(Value::as_object)
                    .ok_or_else(|| bad(&index, "index has no weight_map"))?;
                let mut shard_set: Vec<String> = Vec::new();
                let mut map = BTreeMap::new();
                for (name, shard) in wm {
                    let shard = shard
                        .as_str()
                        .ok_or_else(|| bad(&index, &format!("weight_map[{name}] not a string")))?;
                    let idx = match shard_set.iter().position(|s| s == shard) {
                        Some(i) => i,
                        None => {
                            shard_set.push(shard.to_string());
                            shard_set.len() - 1
                        }
                    };
                    map.insert(name.clone(), idx);
                }
                shard_set.sort();
                // Re-derive indices against the sorted order.
                let mut map2 = BTreeMap::new();
                for (name, shard) in wm {
                    let idx = shard_set
                        .iter()
                        .position(|s| s == shard.as_str().unwrap())
                        .unwrap();
                    map2.insert(name.clone(), idx);
                }
                (shard_set, map2)
            } else {
                let mut names: Vec<String> = fs::read_dir(dir)?
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect();
                if names.is_empty() {
                    return Err(bad(dir, "no *.safetensors and no index (not an hf checkpoint?)"));
                }
                names.sort();
                (names.clone(), BTreeMap::new())
            };
        let mut shards = Vec::new();
        for name in &shard_names {
            let path = dir.join(name);
            let f = fs::File::open(&path).map_err(|e| bad(&path, &format!("open: {e}")))?;
            let mmap = unsafe { Mmap::map(&f) }.map_err(|e| bad(&path, &format!("mmap: {e}")))?;
            let (_, tensors) = parse_shard_header(&mmap, &path)?;
            shards.push(Shard { path, mmap, tensors });
        }
        let mut weight_map = weight_map;
        if weight_map.is_empty() {
            for (si, sh) in shards.iter().enumerate() {
                for t in &sh.tensors {
                    weight_map.insert(t.name.clone(), si);
                }
            }
        }
        Ok(Checkpoint { dir: dir.to_path_buf(), shards, weight_map })
    }

    pub fn tensor_bytes(&self, si: usize, t: &TensorEntry) -> &[u8] {
        &self.shards[si].mmap[t.start as usize..t.end as usize]
    }
}

/// The per-run outcome the census + write produce.
pub struct RunOutcome {
    pub rows: Vec<CensusRow>,
    pub band_c_kept: usize,
    pub n_cast: usize,
    pub wrote_artifact: bool,
}

/// Scan one bf16 tensor: (absmax, subnormal-path count).
fn scan_bf16(bytes: &[u8]) -> (f64, u64) {
    let mut absmax: f32 = 0.0;
    let mut n_sub: u64 = 0;
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let v = bf16_to_f32(bits).abs();
        if v > absmax {
            absmax = v;
        }
        if is_subnormal_path(bits) {
            n_sub += 1;
        }
    }
    (absmax as f64, n_sub)
}

fn row_absmaxes(bytes: &[u8], rows: usize, in_dim: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut m: f32 = 0.0;
        for chunk in bytes[r * in_dim * 2..(r + 1) * in_dim * 2].chunks_exact(2) {
            let v = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])).abs();
            if v > m {
                m = v;
            }
        }
        out.push(m as f64);
    }
    out
}

/// Post-Hadamard absmax over the transformed row prefixes (T5 planning).
fn post_hadamard_absmax(bytes: &[u8], rows: usize, in_dim: usize) -> (f64, usize) {
    let d = largest_pow2_le(in_dim);
    let mut m: f32 = 0.0;
    let mut row = vec![0f32; d];
    for r in 0..rows {
        for (j, x) in row.iter_mut().enumerate() {
            *x = bf16_to_f32(u16::from_le_bytes([
                bytes[(r * in_dim + j) * 2],
                bytes[(r * in_dim + j) * 2 + 1],
            ]));
        }
        walsh_hadamard_row(&mut row);
        for &x in &row {
            let a = x.abs();
            if a > m {
                m = a;
            }
        }
    }
    (m as f64, d)
}

/// The data pass, ported from gguf-recast's cast_pass2: fast path table
/// cast with an Inf guard; slow path materialize, Hadamard, divide by
/// scale, cast. Inf is a halt-class error, never a silent clamp.
pub fn cast_bytes(
    name: &str,
    bytes: &[u8],
    scale: f64,
    hadamard_d: usize,
    in_dim: usize,
) -> io::Result<Vec<u8>> {
    let inf_err = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: produced Inf in the data-pass cast (scale={scale}, hadamard_d={hadamard_d})"),
        )
    };
    let mut out = Vec::with_capacity(bytes.len());
    if hadamard_d == 0 && scale == 1.0 {
        let table = bf16_to_f16_table();
        for chunk in bytes.chunks_exact(2) {
            let h = table[u16::from_le_bytes([chunk[0], chunk[1]]) as usize];
            if h & 0x7fff == 0x7c00 {
                return Err(inf_err());
            }
            out.extend_from_slice(&h.to_le_bytes());
        }
        return Ok(out);
    }
    let mut f: Vec<f32> =
        bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    if hadamard_d != 0 && in_dim > 0 {
        let rows = f.len() / in_dim;
        for r in 0..rows {
            walsh_hadamard_row(&mut f[r * in_dim..r * in_dim + hadamard_d]);
        }
    }
    let scale32 = scale as f32;
    if scale != 1.0 {
        for x in &mut f {
            *x /= scale32;
        }
    }
    for x in &f {
        let h = f32_to_f16_bits(*x);
        if h & 0x7fff == 0x7c00 {
            return Err(inf_err());
        }
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

fn write_shard(
    out_path: &Path,
    tensors: &[(String, String, Vec<usize>, Vec<u8>)], // (name, dtype, shape, data)
    metadata: Option<&BTreeMap<String, String>>,
) -> io::Result<()> {
    let mut header: BTreeMap<String, Value> = BTreeMap::new();
    let mut offset = 0usize;
    for (name, dtype, shape, data) in tensors {
        let end = offset + data.len();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    if let Some(md) = metadata {
        if !md.is_empty() {
            header.insert("__metadata__".into(), serde_json::to_value(md)?);
        }
    }
    let mut hdr = serde_json::to_string(&header)?;
    // Pad the header with spaces to 8-byte alignment (the reference
    // writer's convention; any reader accepts it, and the padding makes
    // the layout deterministic).
    while (8 + hdr.len()) % 8 != 0 {
        hdr.push(' ');
    }
    let tmp = out_path.with_extension("data.tmp");
    {
        let mut w = BufWriter::with_capacity(1 << 22, fs::File::create(&tmp)?);
        for (_, _, _, data) in tensors {
            w.write_all(data)?;
        }
        w.flush()?;
    }
    let mut out = BufWriter::with_capacity(1 << 22, fs::File::create(out_path)?);
    out.write_all(&((hdr.len()) as u64).to_le_bytes())?;
    out.write_all(hdr.as_bytes())?;
    let mut r = fs::File::open(&tmp)?;
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    fs::remove_file(&tmp)?;
    Ok(())
}

/// Census (and, unless dry-run, recast-write) one checkpoint. Returns the
/// outcome; the caller maps it onto exit codes.
pub fn run(
    ckpt: &Checkpoint,
    out_dir: Option<&Path>,
    tier: Tier,
    policy: &crate::policy::Policy,
    force_rescale: bool,
) -> io::Result<RunOutcome> {
    let dry = tier == Tier::DryRun;
    if !dry && out_dir.is_none() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--output is required unless --tier dry-run"));
    }
    let mut rows: Vec<CensusRow> = Vec::new();
    let mut n_cast = 0usize;
    let mut t4_metadata: BTreeMap<String, String> = BTreeMap::new();

    // Per shard: census + collect output tensors.
    let mut shard_outputs: Vec<Vec<(String, String, Vec<usize>, Vec<u8>)>> = Vec::new();
    for shard in &ckpt.shards {
        let mut outs: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        for t in &shard.tensors {
            let bytes = &shard.mmap[t.start as usize..t.end as usize];
            let preserved = policy.matches_preserve(&t.name);
            if t.dtype != "BF16" {
                rows.push(CensusRow {
                    name: t.name.clone(),
                    src_type: t.dtype.clone(),
                    out_type: t.dtype.clone(),
                    band: None,
                    absmax: 0.0,
                    n_elem: t.n_elem(),
                    note: "passthrough".into(),
                    n_subnormal: 0,
                });
                if !dry {
                    outs.push((t.name.clone(), t.dtype.clone(), t.shape.clone(), bytes.to_vec()));
                }
                continue;
            }
            let (absmax, n_sub) = scan_bf16(bytes);
            let band = classify_band(absmax);
            let is_2d = t.shape.len() == 2;
            let (row_abs, post_abs, side) = match tier {
                Tier::T4 if is_2d && !preserved => {
                    let in_dim = t.shape[t.shape.len() - 1];
                    let rows_n = t.n_elem() / in_dim.max(1);
                    (Some(row_absmaxes(bytes, rows_n, in_dim)), None, 0)
                }
                Tier::T5 if is_2d && !preserved => {
                    let in_dim = t.shape[t.shape.len() - 1];
                    let rows_n = t.n_elem() / in_dim.max(1);
                    let (post, d) = post_hadamard_absmax(bytes, rows_n, in_dim);
                    (None, Some(post), d)
                }
                _ => (None, None, 0),
            };
            let action = dispatch(
                tier,
                band,
                absmax,
                is_2d,
                force_rescale,
                preserved,
                row_abs.as_deref(),
                post_abs,
                side,
            );
            let (out_type, note) = match &action {
                Action::KeepBf16 { note } => ("BF16", note.clone()),
                Action::CastF16 { note, .. } => ("F16", note.clone()),
                Action::CastF16RecordRowScales { note, row_scales } => {
                    if !dry {
                        t4_metadata.insert(
                            format!("recast.t4.{}", t.name),
                            serde_json::to_string(&row_scales.iter().map(|&x| x as f32).collect::<Vec<_>>())?,
                        );
                    }
                    ("F16", note.clone())
                }
            };
            if !matches!(action, Action::KeepBf16 { .. }) {
                n_cast += 1;
            }
            rows.push(CensusRow {
                name: t.name.clone(),
                src_type: "BF16".into(),
                out_type: out_type.into(),
                band: Some(band),
                absmax,
                n_elem: t.n_elem(),
                note,
                n_subnormal: n_sub,
            });
            if !dry {
                let data = match &action {
                    Action::KeepBf16 { .. } => bytes.to_vec(),
                    Action::CastF16 { scale, hadamard_d, .. } => {
                        let in_dim = t.shape.last().copied().unwrap_or(0);
                        cast_bytes(&t.name, bytes, *scale, *hadamard_d, in_dim)?
                    }
                    Action::CastF16RecordRowScales { .. } => {
                        cast_bytes(&t.name, bytes, 1.0, 0, 0)?
                    }
                };
                outs.push((t.name.clone(), out_type.into(), t.shape.clone(), data));
            }
        }
        shard_outputs.push(outs);
    }

    let wrote = !dry;
    if let Some(out) = out_dir {
        fs::create_dir_all(out)?;
        for (shard, outs) in ckpt.shards.iter().zip(&shard_outputs) {
            let name = shard.path.file_name().unwrap().to_str().unwrap().to_string();
            let md = if t4_metadata.is_empty() { None } else { Some(&t4_metadata) };
            write_shard(&out.join(&name), outs, md)?;
        }
        // Non-safetensors files verbatim, except config.json's dtype.
        for entry in fs::read_dir(&ckpt.dir)? {
            let entry = entry?;
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.ends_with(".safetensors") || name == "model.safetensors.index.json" {
                // The index is copied verbatim below (the tensor -> shard
                // map is invariant); shards were rewritten above.
                if name == "model.safetensors.index.json" {
                    fs::copy(&p, out.join(&name))?;
                }
                continue;
            }
            if name == "config.json" {
                let text = fs::read_to_string(&p)?;
                let rewritten = rewrite_torch_dtype(&text)
                    .ok_or_else(|| io::Error::new(
                        io::ErrorKind::InvalidData,
                        "config.json: no \"torch_dtype\": \"bfloat16\" to rewrite but tensors were cast",
                    ))?;
                fs::write(out.join(&name), rewritten)?;
                continue;
            }
            fs::copy(&p, out.join(&name))?;
        }
    }
    // The spec law (rules T1Dispatch + ExitCodeSignalsDeviation):
    // band_c_kept counts band-C census rows that were NOT cast at the
    // acting tier -- kept by the T1 fallback or a policy preserve. At
    // T2+ band C casts (rescaled) and is not a deviation.
    let band_c_kept = rows
        .iter()
        .filter(|r| r.band == Some(Band::C) && !dry && r.out_type == "BF16")
        .count();
    Ok(RunOutcome { rows, band_c_kept, n_cast, wrote_artifact: wrote })
}

/// Surgical dtype rewrite: only the dtype assignment changes, the rest of
/// the file stays byte-identical. Handles both key spellings: `dtype`
/// (transformers v5 config schema) and `torch_dtype` (v4 and earlier);
/// an already-float16 checkpoint is a no-op success.
fn rewrite_torch_dtype(text: &str) -> Option<String> {
    let already = regex::Regex::new(r#""(?:torch_)?dtype"\s*:\s*"float16""#).ok()?;
    if already.is_match(text) {
        return Some(text.to_string());
    }
    let re = regex::Regex::new(r#""(torch_)?dtype"(\s*:\s*)"bfloat16""#).ok()?;
    let cow = re.replacen(text, 1, r#""${1}dtype"${2}"float16""#);
    if cow.as_ref() != text {
        Some(cow.into_owned())
    } else {
        None
    }
}

/// The census TSV: gguf-recast's seven columns + n_subnormal appended
/// (the StableCensusSchema law: trailing columns only).
pub fn tsv(rows: &[CensusRow]) -> String {
    let mut s = String::from("name\tsrc_type\tout_type\tband\tabsmax\tn_elem\tnote\tn_subnormal\n");
    for r in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.name,
            r.src_type,
            r.out_type,
            r.band.map(|b| b.as_str().to_string()).unwrap_or_else(|| "-".into()),
            fmt_absmax(r.absmax),
            r.n_elem,
            r.note,
            r.n_subnormal,
        ));
    }
    s
}

fn fmt_absmax(a: f64) -> String {
    if a == 0.0 {
        "0".into()
    } else {
        format!("{a:.6e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_is_surgical() {
        // v4 key, no space
        let cfg = r#"{"architectures":["Q"],"torch_dtype":"bfloat16","vocab_size":1}"#;
        assert_eq!(
            rewrite_torch_dtype(cfg).unwrap(),
            r#"{"architectures":["Q"],"torch_dtype":"float16","vocab_size":1}"#
        );
        // v5 key (transformers 5.x renamed it), with space
        let v5 = r#"{"architectures":["Q"],"dtype": "bfloat16"}"#;
        assert_eq!(
            rewrite_torch_dtype(v5).unwrap(),
            r#"{"architectures":["Q"],"dtype": "float16"}"#
        );
        let already = r#"{"torch_dtype": "float16"}"#;
        assert_eq!(rewrite_torch_dtype(already).unwrap(), already);
        let already5 = r#"{"dtype": "float16"}"#;
        assert_eq!(rewrite_torch_dtype(already5).unwrap(), already5);
        assert!(rewrite_torch_dtype(r#"{"a":1}"#).is_none());
    }
}
