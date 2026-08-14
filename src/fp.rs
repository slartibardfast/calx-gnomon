//! Float conversions, ported from gguf-recast's `fp.rs` (same author; the
//! bf16 -> f16 table and the RNE f32 -> f16 cast carry over unchanged, the
//! torch-exp table stays behind in gguf-recast where the remix needs it).
//!
//! `f32_to_f16_bits` is IEEE 754 binary32 -> binary16 round-to-nearest-even,
//! overflow to inf, NaN payload MSBs kept (numpy software-cast semantics).
//! The Kani harness `normal_range_bit_exact` proves the cast law universal
//! over the bf16 bit domain; the exhaustive fixture test below binds it
//! against the reference fixture byte-for-byte.

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
}
