//! calx-gnomon: the absmax-aware BF16 -> FP16 selective recast, plus the
//! GGUF-side repack tier (AutoRound INT4 checkpoint -> Q4_0/Q4_0_AR16 GGUF
//! with the optional mtp splice, call/0039 P3).
//!
//! Subcommands:
//!   calx-gnomon repack --model-dir <AR-export> --outfile <out.gguf>
//!                      [--splice <fp16-checkpoint-dir>]
//!   (otherwise) the recast gate CLI below.
//!
//! Exit codes (the Wicket law, specs/recast.allium):
//!   0  clean (dry-run census, or a completed run with no band-C tensors)
//!   2  completed with kept band-C tensors (the artifact is valid; the
//!      deviation is signalled -- pipelines halt or acknowledge)
//!   1  halt-class failure (IO, malformed artifact, bad policy/args)

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("repack") {
        return ExitCode::from(run_repack(argv) as u8);
    }
    let code = run(argv);
    ExitCode::from(code as u8)
}

fn repack_usage() -> String {
    "usage: calx-gnomon repack --model-dir <auto-round-export-dir> \
     --outfile <out.gguf> [--splice <fp16-checkpoint-dir>]"
        .into()
}

fn run_repack(argv: Vec<String>) -> i32 {
    let get_opt = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let (Some(model_dir), Some(outfile)) = (get_opt("--model-dir"), get_opt("--outfile")) else {
        eprintln!("calx-gnomon: repack needs --model-dir and --outfile\n{}", repack_usage());
        return 1;
    };
    let splice = get_opt("--splice");
    let result = match &splice {
        Some(dir) => calx_gnomon::gguf::run_with_splice(
            &PathBuf::from(model_dir),
            &PathBuf::from(dir),
            &PathBuf::from(outfile),
        ),
        None => calx_gnomon::gguf::run(&PathBuf::from(model_dir), &PathBuf::from(outfile)),
    };
    match result {
        Ok(()) => {
            eprintln!(
                "calx-gnomon: repack{} complete",
                if splice.is_some() { " (spliced)" } else { "" }
            );
            0
        }
        Err(e) => {
            eprintln!("calx-gnomon: repack failed: {e}");
            1
        }
    }
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
