PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all pipeline paper clean

all: pipeline paper

pipeline:
	./scripts/run_all.sh

paper:
	mkdir -p dist
	cd paper && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=../dist entrega_final.tex
	cp dist/entrega_final.pdf paper/entrega_final.pdf

clean:
	rm -rf build dist tmp
	find paper -maxdepth 1 \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.synctex.gz' -o -name 'missfont.log' \) -delete
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
