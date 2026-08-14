//! The policy file: regex lists naming tensors the recast must not touch.
//! Same shape as gguf-recast's (and the retired Python's): a yaml mapping
//! of lists; `cast_fp16` is parsed for validation parity and, like there,
//! never consulted (the tier dispatch decides casts).

use std::io;
use std::path::Path;

pub struct Policy {
    pub preserve_bf16: Vec<regex::Regex>,
}

impl Policy {
    pub fn load(path: &Path) -> io::Result<Policy> {
        let raw = std::fs::read_to_string(path)?;
        let v: serde_yaml::Value = serde_yaml::from_str(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("policy: {e}")))?;
        let list = |key: &str, default: &[&str]| -> io::Result<Vec<regex::Regex>> {
            let pats: Vec<String> = match v.get(key) {
                Some(serde_yaml::Value::Sequence(s)) => s
                    .iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect(),
                None => default.iter().map(|s| s.to_string()).collect(),
                Some(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("policy key {key} is not a list"),
                    ))
                }
            };
            pats.iter()
                .map(|p| {
                    regex::Regex::new(p).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("policy regex: {e}"))
                    })
                })
                .collect()
        };
        let _ = list("cast_fp16", &[".*"])?;
        Ok(Policy { preserve_bf16: list("preserve_bf16", &[])? })
    }

    pub fn matches_preserve(&self, name: &str) -> bool {
        self.preserve_bf16.iter().any(|p| p.is_match(name))
    }
}

/// The empty policy: recast everything the tier dispatch allows.
impl Default for Policy {
    fn default() -> Self {
        Policy { preserve_bf16: Vec::new() }
    }
}
