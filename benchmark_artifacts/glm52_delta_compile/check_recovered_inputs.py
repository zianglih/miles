#!/usr/bin/env python3
"""Check the sealed input inventory without rereading model-sized payloads.

This checks unchanged filesystem metadata against the post-hash recovery seal;
it is not a fresh content-hash pass. Rehash explicitly if metadata changes.
"""

import argparse
import hashlib
import json
from pathlib import Path


READY_SHA256 = '87dc3807a42f77a53e45a2611a8a436f7a12613ffcb51472a4f5bf1fde7823f8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/hai-workspace/glm52-delta'))
    parser.add_argument('--marker-sha256', default=READY_SHA256)
    args = parser.parse_args()
    root = args.root.resolve()
    marker_path = root / 'artifacts/recovery-inputs-ready.json'
    raw = marker_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.marker_sha256:
        raise ValueError('Readiness marker changed')
    marker = json.loads(raw)
    if marker['status'] != 'ready' or marker['root'] != str(root):
        raise ValueError('Readiness marker belongs to another root or is incomplete')
    for relative, item in marker['inputs'].items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Prepared file missing or replaced: {relative}')
        value = path.stat()
        actual = {key: getattr(value, key) for key in item['local_stat']}
        if actual != item['local_stat']:
            raise ValueError(f'Prepared file metadata changed; rehash before use: {relative}')
    for relative, item in marker['evidence'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Recovery evidence changed: {relative}')
    print(json.dumps({'status': 'ready', 'prepared_files': len(marker['inputs']),
                      'bytes': sum(item['bytes'] for item in marker['inputs'].values()),
                      'marker_sha256': args.marker_sha256, 'check': 'metadata unchanged since full hashing'}))


if __name__ == '__main__':
    main()
