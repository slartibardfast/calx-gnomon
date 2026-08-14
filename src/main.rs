//! calx-gnomon: the absmax-aware BF16 -> FP16 selective recast.
//!
//! Exit codes (the Wicket law, specs/recast.allium):
//!   0  clean (dry-run census, or a completed run with no band-C tensors)
//!   2  completed with kept band-C tensors (the artifact is valid; the
//!      deviation is signalled -- pipelines halt or acknowledge)
//!   1  halt-class failure (IO, malformed artifact, bad policy/args)

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let code = run(std::env::args().skip(1).collect());
    ExitCode::from(code as u8)
}

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    tier: String,
    tsv: Option<PathBuf>,
    policy: Option<PathBuf>,
    force_rescale: bool,
}

fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut tier = None;
    let mut tsv = None;
    let mut policy = None;
    let mut force_rescale = false;
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let mut take = |name: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg {
            "--input" => input = Some(PathBuf::from(take("--input")?)),
            "--output" => output = Some(PathBuf::from(take("--output")?)),
            "--tier" => tier = Some(take("--tier")?),
            "--tsv" => tsv = Some(PathBuf::from(take("--tsv")?)),
            "--policy" => policy = Some(PathBuf::from(take("--policy")?)),
            "--force-rescale" => force_rescale = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 1;
    }
    let input = input.ok_or("--input is required")?;
    let tier = tier.ok_or("--tier is required (dry-run|t1|t2|t3|t4|t5)")?;
    if calx_gnomon::Tier::parse(&tier).is_none() {
        return Err(format!("unknown tier {tier:?}"));
    }
    Ok(Args { input, output, tier, tsv, policy, force_rescale })
}

fn usage() -> String {
    "usage: calx-gnomon --input <hf-checkpoint-dir> --tier <dry-run|t1|t2|t3|t4|t5> \
     [--output <dir>] [--tsv <file>] [--policy <file>] [--force-rescale]"
        .into()
}

fn run(argv: Vec<String>) -> i32 {
    let args = match parse_args(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("calx-gnomon: {e}\n{}", usage());
            return 1;
        }
    };
    let tier = calx_gnomon::Tier::parse(&args.tier).unwrap();
    let policy = match &args.policy {
        Some(p) => match calx_gnomon::policy::Policy::load(p) {
            Ok(pol) => pol,
            Err(e) => {
                eprintln!("calx-gnomon: {e}");
                return 1;
            }
        },
        None => calx_gnomon::policy::Policy::default(),
    };
    let ckpt = match calx_gnomon::hf::Checkpoint::open(&args.input) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("calx-gnomon: {}", e);
            return 1;
        }
    };
    let outcome = match calx_gnomon::hf::run(&ckpt, args.output.as_deref(), tier, &policy, args.force_rescale) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("calx-gnomon: {e}");
            return 1;
        }
    };
    let tsv_text = calx_gnomon::hf::tsv(&outcome.rows);
    match &args.tsv {
        Some(p) => {
            if let Err(e) = std::fs::write(p, &tsv_text) {
                eprintln!("calx-gnomon: --tsv {}: {e}", p.display());
                return 1;
            }
        }
        None => print!("{tsv_text}"),
    }
    let n_a = outcome.rows.iter().filter(|r| r.band == Some(calx_gnomon::Band::A)).count();
    let n_b = outcome.rows.iter().filter(|r| r.band == Some(calx_gnomon::Band::B)).count();
    let n_c = outcome.rows.iter().filter(|r| r.band == Some(calx_gnomon::Band::C)).count();
    eprintln!(
        "calx-gnomon: {} tensors (A {n_a} / B {n_b} / C {n_c}), cast {}, band-C kept {}",
        outcome.rows.len(),
        outcome.n_cast,
        outcome.band_c_kept
    );
    if tier == calx_gnomon::Tier::DryRun {
        return 0; // rule DryRunWritesNothing: exit 0, no artifact
    }
    if outcome.band_c_kept > 0 {
        2
    } else {
        0
    }
}
