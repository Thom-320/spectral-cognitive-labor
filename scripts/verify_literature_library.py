#!/usr/bin/env python3
"""Comprueba una copia local de la biblioteca contra docs/literature/paper_catalog.json.

Los PDF no se versionan en este repositorio. Este script verifica que una carpeta
local contiene exactamente los archivos catalogados y que sus SHA-256 coinciden.

Uso: python scripts/verify_literature_library.py --dir ~/Downloads/Spectral_SODCL_papers
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'docs/literature/paper_catalog.json'


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True, help='Carpeta con los PDF')
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())

    ok, missing, mismatched = [], [], []
    for entry in catalog:
        path = args.dir.expanduser() / entry['file']
        if not path.exists():
            missing.append(entry['file'])
        elif sha256(path) != entry['sha256']:
            mismatched.append(entry['file'])
        else:
            ok.append(entry['file'])

    catalogued = {e['file'] for e in catalog}
    extra = sorted(p.name for p in args.dir.expanduser().glob('*.pdf') if not p.name.startswith('._') and p.name not in catalogued)

    print(f'Catálogo: {len(catalog)} entradas. Verificados: {len(ok)}.')
    for label, items in [('Faltan', missing), ('Hash distinto', mismatched), ('Sin catalogar', extra)]:
        if items:
            print(f'{label}: ' + ', '.join(items))
    return 0 if not missing and not mismatched else 1


if __name__ == '__main__':
    sys.exit(main())
