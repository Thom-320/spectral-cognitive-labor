# Reproducibility Notes

This project is designed to be run from the repository root.

## Environment

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To rebuild the PDF, a LaTeX installation with `pdflatex` and `latexmk` is
required.

## Full Analysis

Run:

```bash
make pipeline
```

This executes the eight analysis scripts in order and regenerates the main CSV,
NPZ, and PNG outputs under `data/results/` and `figures/`.

## One-Command Rebuild

Run:

```bash
make all
```

This runs the complete analysis pipeline.

## Course Submission Artifact

The Spanish course submission PDF is maintained in the local submission package,
not in this public portfolio repository. The public repository is intended to
reproduce the analysis tables and figures used by that report.
