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

## Final Paper

Run:

```bash
make paper
```

This compiles `paper/entrega_final.tex` and copies the final PDF to
`paper/entrega_final.pdf`.

## One-Command Rebuild

Run:

```bash
make all
```

This runs the analysis pipeline and rebuilds the final PDF.

## Expected Final Artifact

The course submission PDF is:

```text
paper/entrega_final.pdf
```

The source used to generate it is:

```text
paper/entrega_final.tex
```
