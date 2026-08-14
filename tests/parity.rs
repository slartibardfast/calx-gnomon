//! The HF-side parity oracle (contract SelectiveRecast.hf_parity):
//! bitwise equality with torch's own bf16 -> f16 cast over the full bf16
//! bit domain.
//!
//! Opt-in because the fixture needs the reference venv: set
//! CALX_GNOMON_TORCH_FIXTURE to a file of little-endian (u16 bf16, u16
//! f16) pairs covering the 65536-pattern domain (generate with the
//! plan/0147 BUILD.lock venv:
//!   python -c "import torch,struct;
//!     t=torch.arange(65536,dtype=torch.int32).view(torch.uint16)...").
//! Without the fixture the test SKIPS (reported, not silently passed);
//! CI runs it whenever the fixture is committed.

mod common;

use calx_gnomon::fp::bf16_to_f16_table;

#[test]
fn torch_parity_bitwise() {
    let path = match std::env::var("CALX_GNOMON_TORCH_FIXTURE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: CALX_GNOMON_TORCH_FIXTURE not set (no reference fixture)");
            return;
        }
    };
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 65536 * 4, "fixture must cover the full bf16 domain");
    let table = bf16_to_f16_table();
    let mut mismatches = 0usize;
    for i in 0..65536 {
        let bf = u16::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1]]);
        let f16 = u16::from_le_bytes([bytes[i * 4 + 2], bytes[i * 4 + 3]]);
        if table[bf as usize] != f16 {
            if mismatches < 8 {
                eprintln!("mismatch: bf16 {bf:#06x} ours {:#06x} torch {f16:#06x}", table[bf as usize]);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} of 65536 patterns differ from torch");
}
