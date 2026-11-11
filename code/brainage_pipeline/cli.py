"""Single entry point for preparation, evaluation and reproducible artifacts."""
import argparse
import json
from pathlib import Path
from foldwise_dwi import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--root", type=Path, default=ROOT,
                   help="Data root containing data/final and results/final input manifests")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--min-cn", type=int, default=5)
    q.add_argument("--age-tolerance", type=float, default=0.01)
    q = sub.add_parser("train")
    q.add_argument("--prepared", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/default.json"
    )
    q.add_argument("--population", choices=["paired", "DWI", "fMRI"], default="paired")
    q.add_argument("--resume", action="store_true")
    q.add_argument("--fraction", type=float, default=1.0)
    q = sub.add_parser("audit")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--allow-partial", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("slides")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--prepared", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("clinical")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--spec", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("embeddings")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--clinical-spec", type=Path)
    q = sub.add_parser("attribute")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--task", type=Path, required=True)
    q.add_argument("--prepared", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--roi-labels", type=Path)
    q = sub.add_parser("learning-curves")
    q.add_argument("--runs", type=Path, nargs="+", required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("diagnostics")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--prepared", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare

        print(
            prepare(args.root.resolve(), args.output, args.min_cn, args.age_tolerance).to_string(
                index=False
            )
        )
    elif args.command == "train":
        from .engine import run

        settings = json.loads(args.config.read_text())
        validate_config(settings)
        run(
            args.prepared,
            args.output,
            settings,
            args.population,
            args.resume,
            args.fraction,
        )
    elif args.command == "audit":
        from .audit import audit

        print(json.dumps(audit(args.run, args.allow_partial), indent=2))
    elif args.command == "report":
        from .reporting import report

        report(args.run, args.output)
    elif args.command == "slides":
        from .slides import slides

        slides(args.run, args.prepared, args.output)
    elif args.command == "clinical":
        from .clinical import clinical

        clinical(args.run, args.spec, args.output)
    elif args.command == "embeddings":
        from .downstream import embeddings

        embeddings(args.run, args.output, args.clinical_spec)
    elif args.command == "attribute":
        from .downstream import attribute

        attribute(args.run, args.task, args.prepared, args.output, args.roi_labels)
    elif args.command == "learning-curves":
        from .downstream import learning_curves

        learning_curves(args.runs, args.output)
    elif args.command == "diagnostics":
        from .diagnostics import diagnostics

        diagnostics(args.run, args.prepared, args.output)


def validate_config(s):
    if min(s["outer_folds"], s["inner_folds"]) < 2:
        raise ValueError("Need >=2 folds")
    if set(s["learners"]) - {"ridge", "gine"}:
        raise ValueError("Unknown learner")
    if set(s["harmonization_modes"]) - {"none", "batch", "age_mean"}:
        raise ValueError("Unknown harmonization mode")
    if not s["ridge_alphas"] or min(s["ridge_alphas"]) <= 0:
        raise ValueError("Positive Ridge alphas required")
    for c in s["gine_grid"]:
        if c["layers"] < 1 or c["width"] < 2 or not 0 < c["threshold"] <= 1:
            raise ValueError("Invalid graph architecture/sparsity")
    if s["training"]["max_epochs"] < 1 or s["training"]["patience"] < 1:
        raise ValueError("Invalid epoch budget")
