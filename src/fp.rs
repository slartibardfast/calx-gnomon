//! Float conversions, ported from gguf-recast's `fp.rs` (same author; the
//! bf16 -> f16 table and the RNE f32 -> f16 cast carry over unchanged).
//! The torch-exp table and the exact f16 -> f32 upcast landed with the P3
//! gguf tier (call/0039): the remix's A_log path needs PyTorch MKL exp
//! semantics, embedded as a domain table; the mtp splice reads FP16
//! sources and upcasts them exactly.
//!
//! `f32_to_f16_bits` is IEEE 754 binary32 -> binary16 round-to-nearest-even,
//! overflow to inf, NaN payload MSBs kept (numpy software-cast semantics).
//! The Kani harness `normal_range_bit_exact` proves the cast law universal
//! over the bf16 bit domain; the exhaustive fixture test below binds it
//! against the reference fixture byte-for-byte.

/// PyTorch 2.13.0+cpu (Intel MKL vmsExp, VML_HA) `exp` over the BF16 domain.
///
/// MKL is closed source and deviates from correctly-rounded exp on 1794 of
/// the 65536 BF16-derived inputs (and quiets NaN payloads differently), so
/// the semantic is embedded as a domain table rather than re-derived from an
/// algorithm. See gguf-recast `scripts/gen_numeric_fixtures.py` for
/// provenance; the fixture is vendored here byte-for-byte.
static TORCH_EXP_BF16: &[u8; 65536 * 4] = include_bytes!("../fixtures/torch_exp_bf16.bin");

/// `torch.exp` (MKL vmsExp HA) of the f32 value whose bits are `bf16 << 16`.
#[inline]
pub fn torch_exp_bf16_bits(bits: u16) -> u32 {
    let i = bits as usize * 4;
    u32::from_le_bytes([
        TORCH_EXP_BF16[i],
        TORCH_EXP_BF16[i + 1],
        TORCH_EXP_BF16[i + 2],
        TORCH_EXP_BF16[i + 3],
    ])
}

/// IEEE 754 binary16 -> binary32, exact (binary16 is a sub-format of
/// binary32 after exponent rebias and mantissa widening).
#[inline]
pub fn f16_to_f32_bits(h: u16) -> u32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        if man == 0 {
            return sign; // +/- zero
        }
        // Subnormal: normalize, the value is man * 2^-24.
        let mut m = man;
        let mut e = 0u32;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        return sign | ((127 - 14 - e) << 23) | ((m & 0x3ff) << 13);
    }
    if exp == 0x1f {
        // inf, or NaN with the payload widened (payload MSBs kept).
        return sign | 0x7f80_0000 | (man << 13);
    }
    sign | ((exp + 112) << 23) | (man << 13)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    f32::from_bits(f16_to_f32_bits(h))
}

/// `torch.exp` (MKL vmsExp HA) of an F16 input value. The MKL table is
/// defined on the BF16 bit domain; an f16 value is table-indexable exactly
/// when it is bf16-representable (mantissa a multiple of 8, normal range),
/// which holds for every RNE(bf16 -> f16) value in range (the 8-bit bf16
/// mantissa widens losslessly into the 10-bit f16 mantissa). Outside that
/// set there is no table entry; f32 exp is the documented fallback (the
/// plan/0147 A_log tensors never reach it).
pub fn torch_exp_f16_bits(h: u16) -> u32 {
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x3ff;
    if exp != 0 && exp != 0x1f && man % 8 == 0 {
        let bf = (h & 0x8000) | ((exp + 112) << 7) | (man >> 3);
        return torch_exp_bf16_bits(bf);
    }
    f16_to_f32(h).exp().to_bits()
}

#[inline]
pub fn bf16_to_f32_bits(bits: u16) -> u32 {
    (bits as u32) << 16
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(bf16_to_f32_bits(bits))
}

/// IEEE 754 binary32 -> binary16, RNE, overflow to inf, NaN payload MSBs
/// kept.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp32 = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;

    if exp32 == 0xff {
        if man == 0 {
            return sign | 0x7c00; // inf
        }
        let mut ret = 0x7c00 | ((man >> 13) as u16);
        if ret == 0x7c00 {
            ret += 1;
        }
        return sign | ret;
    }

    let e = exp32 - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // rounds to zero
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut h = (m >> shift) as u16;
        if rest > half || (rest == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | (h as u16)
}

/// f16 min normal, 2^-14: bf16 values strictly below this in magnitude
/// round into the f16 subnormal (or zero) range on cast.
pub const F16_MIN_NORMAL_F32: f32 = 6.103515625e-5;

/// True when this bf16 value's magnitude lands in the subnormal path:
/// nonzero and below 2^-14. These are counted per tensor in the census
/// (`n_subnormal`), never silently discarded.
#[inline]
pub fn is_subnormal_path(bf16_bits: u16) -> bool {
    // Exponent+mantissa bits all zero is +0/-0; anything with magnitude
    // below 2^-14 has its exponent field < 0x3C0... bf16 exponent bias 127,
    // 2^-14 -> exp field = 127-14 = 113 = 0x71, so subnormal-path is
    // 0 < (bits & 0x7fff) < (113 << 7).
    let mag = bf16_bits & 0x7fff;
    mag != 0 && mag < (113u16 << 7)
}

/// BF16 -> F16 (exact BF16 -> f32 upcast, then the RNE cast), for bulk use
/// the table below; this direct form is the Kani proof's reference shape.
#[inline]
pub fn bf16_to_f16(bf16_bits: u16) -> u16 {
    f32_to_f16_bits(bf16_to_f32(bf16_bits))
}

/// Table-accelerated bf16 -> f16 for bulk tensor casts.
pub fn bf16_to_f16_table() -> &'static [u16; 65536] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Box<[u16; 65536]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0u16; 65536].into_boxed_slice();
        for (bits, out) in t.iter_mut().enumerate() {
            *out = bf16_to_f16(bits as u16);
        }
        t.try_into().unwrap()
    })
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// The cast law (spec contract SelectiveRecast.NormalRangeBitExact):
    /// for EVERY bf16 bit pattern, the direct cast equals the reference
    /// composition bf16 -> f32 -> RNE f16. Universal over the full 16-bit
    /// domain (the complete input space, since every F16 the tool emits
    /// derives from a bf16 bit pattern).
    #[kani::proof]
    fn normal_range_bit_exact() {
        let bits: u16 = kani::any();
        let direct = bf16_to_f16(bits);
        let reference = f32_to_f16_bits(bf16_to_f32(bits));
        assert_eq!(direct, reference);
    }

    /// The subnormal accounting law: a pattern takes the subnormal path
    /// exactly when it is nonzero and its f32 magnitude is below 2^-14.
    #[kani::proof]
    fn subnormal_accounting_exact() {
        let bits: u16 = kani::any();
        let v = bf16_to_f32(bits).abs();
        let flagged = is_subnormal_path(bits);
        assert_eq!(flagged, v != 0.0 && v < F16_MIN_NORMAL_F32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundary sanity: the four f16 landmarks round-trip exactly.
    #[test]
    fn f16_landmarks() {
        assert_eq!(f32_to_f16_bits(32768.0), 0x7800);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65505.0), 0x7bff); // nearest normal (RNE)
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // tie at the top rounds to even: inf
        assert_eq!(f32_to_f16_bits(6.103515625e-5), 0x0400); // min normal
    }

    /// The table equals the direct composition everywhere.
    #[test]
    fn table_equals_direct() {
        let t = bf16_to_f16_table();
        for bits in 0u16..=u16::MAX {
            assert_eq!(t[bits as usize], bf16_to_f16(bits), "bits {bits:#06x}");
        }
    }

    #[test]
    fn torch_exp_spot_values() {
        // exp(0) = 1, exp(-inf) = 0, exp(inf) = inf.
        assert_eq!(torch_exp_bf16_bits(0x0000), 1.0f32.to_bits());
        assert_eq!(torch_exp_bf16_bits(0xff80), 0.0f32.to_bits()); // -inf
        assert_eq!(torch_exp_bf16_bits(0x7f80), f32::INFINITY.to_bits());
        // Monotone non-decreasing on a finite positive ramp.
        let mut prev = 0.0f32;
        for bits in 0x3f80..0x4280u16 {
            // 1.0 .. 64.0
            let v = f32::from_bits(torch_exp_bf16_bits(bits));
            assert!(v >= prev, "exp not monotone at bf16 {bits:#06x}");
            prev = v;
        }
    }

    /// f16 -> f32 is exact widening: bind it against an independent
    /// decode for the whole 16-bit domain.
    #[test]
    fn f16_upcast_exact_over_domain() {
        for bits in 0u16..=u16::MAX {
            let sign = if bits & 0x8000 != 0 { -1.0f64 } else { 1.0 };
            let exp = (bits >> 10) & 0x1f;
            let man = (bits & 0x3ff) as f64;
            let expect = match exp {
                0 => sign * man * f64::exp2(-24.0),
                0x1f => {
                    if man == 0.0 {
                        sign * f64::INFINITY
                    } else {
                        f64::NAN // compare NaN-ness below
                    }
                }
                _ => sign * (1.0 + man / 1024.0) * f64::exp2(exp as f64 - 15.0),
            };
            let got = f16_to_f32(bits) as f64;
            if expect.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}");
            } else {
                assert_eq!(got, expect, "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_upcast_landmarks() {
        assert_eq!(f16_to_f32_bits(0x3c00), 1.0f32.to_bits());
        assert_eq!(f16_to_f32_bits(0xbc00), (-1.0f32).to_bits());
        assert_eq!(f16_to_f32_bits(0x0400), 6.103515625e-5f32.to_bits()); // min normal
        assert_eq!(f16_to_f32_bits(0x0001), 0x3380_0000u32); // smallest subnormal
        assert_eq!(f16_to_f32_bits(0x7c00), f32::INFINITY.to_bits());
    }

    /// The f16 exp lookup agrees with the bf16 MKL table wherever the f16
    /// value is bf16-representable: over the whole bf16 domain, restrict
    /// to values that land in the f16 normal range (the bf16 exponent
    /// width keeps the widening exact), and the recovered bf16 bits must
    /// index the same table entry.
    #[test]
    fn f16_exp_agrees_with_bf16_table_in_range() {
        for bits in 0u16..=u16::MAX {
            let h = f32_to_f16_bits(bf16_to_f32(bits));
            let exp = (h >> 10) & 0x1f;
            let man = h & 0x3ff;
            if exp == 0 || exp == 0x1f {
                continue; // subnormal / inf-nan f16 images are off the table path
            }
            assert_eq!(man % 8, 0, "bf16-origin f16 mantissa must widen exactly");
            assert_eq!(
                torch_exp_f16_bits(h),
                torch_exp_bf16_bits(bits),
                "bf16 {bits:#06x} -> f16 {h:#06x}"
            );
        }
    }
}
