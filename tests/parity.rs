//! The HF-side parity oracle (contract SelectiveRecast.hf_parity):
//! bitwise equality with torch's own bf16 -> f16 cast over the FINITE bf16
//! bit domain, plus the boundary law that every divergence is confined to
//! NaN payloads.
//!
//! NaN payloads are a documented semantic difference, not a defect: this
//! tool carries numpy's payload-MSB law (the recipe's, gguf-recast's), torch
//! canonicalizes differently. The spec's band law makes the difference
//! UNREACHABLE in the cast path: a NaN absmax classifies band C, and band C
//! is never cast at scale 1.0 -- so the test proves both that finite inputs
//! agree bitwise and that only NaN inputs may diverge.
//!
//! Opt-in: CALX_GNOMON_TORCH_FIXTURE must point at 65536 little-endian
//! (u16 bf16, u16 f16) pairs covering the domain (the plan/0147 venv
//! generates it). Without the fixture the test SKIPS loudly.

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
    let mut finite_mismatches = 0usize;
    for i in 0..65536usize {
        let bf = u16::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1]]);
        let f16 = u16::from_le_bytes([bytes[i * 4 + 2], bytes[i * 4 + 3]]);
        if table[bf as usize] != f16 {
            mismatches += 1;
            let is_nan_input = (bf & 0x7f80) == 0x7f80 && (bf & 0x007f) != 0;
            if !is_nan_input {
                if finite_mismatches < 8 {
                    eprintln!(
                        "FINITE mismatch: bf16 {bf:#06x} ours {:#06x} torch {f16:#06x}",
                        table[bf as usize]
                    );
                }
                finite_mismatches += 1;
            }
        }
    }
    assert_eq!(finite_mismatches, 0, "{finite_mismatches} finite patterns differ from torch");
    assert!(mismatches <= 254, "divergences beyond the NaN payload set: {mismatches}");
    if mismatches > 0 {
        eprintln!("note: {mismatches} NaN-payload divergences (documented semantic difference, unreachable in the cast path)");
    }
}
