//! calx-gnomon: the absmax-aware BF16 -> FP16 selective recast over both
//! artifact classes (HF safetensors, GGUF), all tiers. Behavioural law:
//! `specs/recast.allium` (call/0039, plan/0146).
//!
//! Core/adapter seam: this lib holds the artifact-generic core (bands,
//! tier dispatch, census records, float laws), the hf adapter, and the
//! gguf-side repack tier (AutoRound INT4 checkpoint -> Q4_0/Q4_0_AR16
//! GGUF with the mtp splice, call/0039 P3). The core never sees a file
//! format; adapters move bytes and call these.

pub mod fp;
pub mod gguf;
pub mod hadamard;
pub mod hf;
pub mod policy;

pub const FP16_MAX: f64 = 65504.0;
pub const FP16_HALF_RANGE: f64 = 32768.0;
pub const RESCALE_TARGET: f64 = 30000.0;
pub const MIN_SCALE: f64 = 1e-30;

/// The band a tensor falls in: a total function of its absmax (NaN
/// classifies band C, the carried Python quirk: NaN fails both <= tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    A,
    B,
    C,
}

impl Band {
    pub fn as_str(self) -> &'static str {
        match self {
            Band::A => "A",
            Band::B => "B",
            Band::C => "C",
        }
    }
}

pub fn classify_band(absmax: f64) -> Band {
    if absmax <= FP16_HALF_RANGE {
        Band::A
    } else if absmax <= FP16_MAX {
        Band::B
    } else {
        Band::C
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Tier {
    DryRun,
    T1,
    T2,
    T3,
    T4,
    T5,
}

impl Tier {
    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "dry-run" => Some(Tier::DryRun),
            "t1" => Some(Tier::T1),
            "t2" => Some(Tier::T2),
            "t3" => Some(Tier::T3),
            "t4" => Some(Tier::T4),
            "t5" => Some(Tier::T5),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::DryRun => "dry-run",
            Tier::T1 => "t1",
            Tier::T2 => "t2",
            Tier::T3 => "t3",
            Tier::T4 => "t4",
            Tier::T5 => "t5",
        }
    }
}

/// One census row: what the gnomon indicates for one tensor. The TSV
/// schema law (surface StableCensusSchema) binds the writer to the
/// gguf-recast seven columns + `n_subnormal` appended.
#[derive(Debug, Clone)]
pub struct CensusRow {
    pub name: String,
    pub src_type: String,
    pub out_type: String,
    /// None for non-bf16 passthrough rows (TSV "-").
    pub band: Option<Band>,
    pub absmax: f64,
    pub n_elem: usize,
    pub note: String,
    pub n_subnormal: u64,
}

/// What the tier dispatch decided for one tensor. The dispatch is pure
/// given (band, dims, tier, force_rescale, policy_preserved, absmax):
/// the Kani lane proves its laws over the band domain; the adapters
/// execute it.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Keep bf16: band C at T1, or policy preserve at any tier.
    KeepBf16 { note: String },
    /// RNE cast at scale 1.0 (T1 A/B; T2-T5 A/B; T4/T5 fallbacks).
    CastF16 { scale: f64, hadamard_d: usize, note: String },
    /// T4: plain cast in the data pass, per-row scales recorded in the
    /// artifact's metadata, never applied (the carried quirk, law in
    /// rule T4RecordsScalesOnly).
    CastF16RecordRowScales { row_scales: Vec<f64>, note: String },
}

/// The pure decision for one tensor, note-free (the Kani lane's proof
/// target; the adapter layer renders notes around it).
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Keep bf16: band C at T1, or policy preserve at any tier.
    Keep,
    /// RNE cast at the given scale (and Hadamard side for T5 2-D).
    Cast { scale: f64, hadamard_d: usize },
    /// T4 2-D: plain cast in the data pass, per-row scales recorded in
    /// the artifact's metadata, never applied (the carried quirk).
    CastRecordRowScales,
}

/// The note-free tier dispatch the proofs bind. Mirrors gguf-recast @
/// 5887dfa's decision structure exactly.
pub fn decide(
    tier: Tier,
    band: Band,
    absmax: f64,
    is_2d: bool,
    force_rescale: bool,
    policy_preserved: bool,
    post_hadamard_absmax: f64,
    hadamard_side: usize,
) -> Decision {
    if policy_preserved {
        return Decision::Keep;
    }
    let t2_scale = if band == Band::C || force_rescale { absmax / RESCALE_TARGET } else { 1.0 };
    match tier {
        Tier::DryRun => Decision::Keep,
        Tier::T1 => match band {
            Band::A | Band::B => Decision::Cast { scale: 1.0, hadamard_d: 0 },
            Band::C => Decision::Keep,
        },
        Tier::T2 | Tier::T3 => Decision::Cast { scale: t2_scale, hadamard_d: 0 },
        Tier::T4 => {
            if !is_2d {
                Decision::Cast { scale: t2_scale, hadamard_d: 0 }
            } else {
                Decision::CastRecordRowScales
            }
        }
        Tier::T5 => {
            if !is_2d {
                Decision::Cast { scale: t2_scale, hadamard_d: 0 }
            } else {
                let scale = if post_hadamard_absmax > FP16_HALF_RANGE || force_rescale {
                    (post_hadamard_absmax / RESCALE_TARGET).max(MIN_SCALE)
                } else {
                    1.0
                };
                Decision::Cast { scale, hadamard_d: hadamard_side }
            }
        }
    }
}

/// The dispatch, ported from gguf-recast @ 5887dfa src/recast.rs, now
/// law in specs/recast.allium: `decide` chooses, `dispatch` renders the
/// census note beside the decision. `is_2d` gates the T4/T5 per-row and
/// Hadamard paths; non-2D falls back to the T2 rescale rule.
pub fn dispatch(
    tier: Tier,
    band: Band,
    absmax: f64,
    is_2d: bool,
    force_rescale: bool,
    policy_preserved: bool,
    row_absmaxes: Option<&[f64]>,
    post_hadamard_absmax: Option<f64>,
    hadamard_side: usize,
) -> Action {
    if policy_preserved {
        return Action::KeepBf16 { note: "policy-preserve".into() };
    }
    let decision = decide(
        tier,
        band,
        absmax,
        is_2d,
        force_rescale,
        policy_preserved,
        post_hadamard_absmax.unwrap_or(absmax),
        hadamard_side,
    );
    match decision {
        Decision::Keep => match tier {
            Tier::DryRun => Action::KeepBf16 { note: format!("dry-run band {}", band.as_str()) },
            _ => Action::KeepBf16 { note: "T1 BF16 fallback".into() },
        },
        Decision::Cast { scale, hadamard_d } => Action::CastF16 {
            scale,
            hadamard_d,
            note: match tier {
                // gguf-recast's exact note formats (differential parity).
                Tier::T1 => format!("T1 RNE-cast (band {})", band.as_str()),
                Tier::T2 | Tier::T3 if hadamard_d == 0 && scale == 1.0 => {
                    format!("{} no-rescale", tier.as_str())
                }
                Tier::T2 | Tier::T3 => format!("{} per-tensor /{}", tier.as_str(), fmt_g6(scale)),
                Tier::T4 => format!("T4 falls-back-T2 /{}", fmt_g6(scale)),
                _ => format!(
                    "T5 hadamard d={hadamard_d}, post_absmax={}, /{}",
                    fmt_g6(post_hadamard_absmax.unwrap_or(absmax)),
                    fmt_g6(scale)
                ),
            },
        },
        Decision::CastRecordRowScales => {
            let rows = row_absmaxes.unwrap_or(&[]);
            let row_scales: Vec<f64> = rows
                .iter()
                .map(|&ra| {
                    if ra > FP16_HALF_RANGE || force_rescale {
                        ra / RESCALE_TARGET
                    } else {
                        1.0
                    }
                })
                .collect();
            let n_scaled = row_scales.iter().filter(|&&s| s != 1.0).count();
            Action::CastF16RecordRowScales {
                row_scales,
                note: format!("T4 per-channel ({n_scaled}/{} rows scaled)", rows.len()),
            }
        }
    }
}
/// Python `f"{x:.6g}"` (C printf %.6g), ported from gguf-recast: the
/// census notes' number format, kept identical for differential parity.
pub fn fmt_g6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{:.5e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let neg = mant.starts_with('-');
    debug_assert_eq!(digits.len(), 6);
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if !(-4..6).contains(&exp) {
        let frac = digits[1..].trim_end_matches('0');
        out.push(digits.as_bytes()[0] as char);
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    } else if exp >= 0 {
        let e = exp as usize;
        out.push_str(&digits[..=e]);
        let frac = digits[e + 1..].trim_end_matches('0');
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        let frac = digits.trim_end_matches('0');
        out.push_str(frac);
    }
    out
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// The band law (invariant BandTotallyClassified): classification is
    /// the threshold chain itself, total over every non-NaN absmax. Kani
    /// proves the chain has no gaps: every f64 lands in exactly one band,
    /// and the boundaries sit at the config constants.
    #[kani::proof]
    fn band_totally_classified() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan());
        let b = classify_band(x);
        match b {
            Band::A => assert!(x <= FP16_HALF_RANGE),
            Band::B => assert!(x > FP16_HALF_RANGE && x <= FP16_MAX),
            Band::C => assert!(x > FP16_MAX),
        }
    }

    /// Monotonicity: a larger absmax never classifies into a stricter
    /// band (the plan/0146 BandClassification obligation).
    #[kani::proof]
    fn band_monotone() {
        let x: f64 = kani::any();
        let y: f64 = kani::any();
        kani::assume(!x.is_nan() && !y.is_nan() && x <= y);
        let rank = |b: Band| match b {
            Band::A => 0,
            Band::B => 1,
            Band::C => 2,
        };
        assert!(rank(classify_band(x)) <= rank(classify_band(y)));
    }

    /// Band C is never cast at scale 1.0 (contract
    /// BandCNeverCastAtScaleOne): at T1 the decision keeps bf16; at
    /// T2/T3 (and the T4/T5 non-2D fallbacks) the scale carries the
    /// rescale absmax/30000, which exceeds 2 for any band-C absmax.
    /// T4 2D is the record-only quirk (rule T4RecordsScalesOnly): the
    /// data pass plain-casts and HALTS on Inf rather than emit one.
    #[kani::proof]
    fn band_c_never_cast_at_scale_one() {
        let tier: Tier = kani::any();
        let absmax: f64 = kani::any();
        let is_2d: bool = kani::any();
        let force: bool = kani::any();
        let pol: bool = kani::any();
        kani::assume(!absmax.is_nan());
        kani::assume(absmax > FP16_MAX);
        let d = decide(tier, Band::C, absmax, is_2d, force, pol, absmax, 1);
        match d {
            Decision::Keep => {}
            Decision::CastRecordRowScales => {}
            Decision::Cast { scale, hadamard_d } => {
                if hadamard_d == 0 {
                    assert!(scale != 1.0);
                }
            }
        }
    }

}
