#!/usr/bin/env python3
"""Seal already hash-verified recovery inputs and preserve their small evidence."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path('/hai-workspace/glm52-delta')
BACKUP = Path('/data/ziangli/glm52-delta-sync-c2/prepared-inputs')
SETUP = Path('/data/ziangli/glm52-delta-sync-c2/setup/recovery-20260926-evidence')
STAMP = '20260926T021916Z-4640'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat_record(path):
    value = path.stat()
    return {key: getattr(value, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError(f'Refuse to overwrite different evidence: {target}')
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())


def main():
    artifacts = ROOT / 'artifacts'
    assert (artifacts / 'recovery-20260926T021828Z-prepare-all.exit').read_text().strip() == '0'
    assert 'RECOVERY_PREPARE_COMPLETE' in (artifacts / 'recovery-20260926T021828Z-prepare-all.log').read_text()
    for stage in ('download-data', 'download-model', 'nvfp4', 'torch-dist'):
        assert (artifacts / f'recovery-{STAMP}-{stage}.exit').read_text().strip() == '0'
    header = artifacts / 'recovery-nvfp4-header-verification.json'
    assert json.loads(header.read_text())['matched_original_headers'] is True
    manifest_path = BACKUP / f'input-backup-{STAMP}.json'
    manifest = json.loads(manifest_path.read_text())
    inputs, inventory = {}, {}
    for relative, entries in manifest['files'].items():
        inventory[relative] = {'files': len(entries), 'bytes': sum(item['bytes'] for item in entries.values())}
        for name, expected in entries.items():
            relative_file = str(Path(relative) / name)
            source, durable = ROOT / relative_file, BACKUP / relative_file
            assert not source.is_symlink() and not durable.is_symlink()
            assert source.stat().st_size == durable.stat().st_size == expected['bytes']
            with durable.open('rb') as stream:
                os.fsync(stream.fileno())
            inputs[relative_file] = dict(expected, local_stat=stat_record(source), durable_stat=stat_record(durable))
    for path in sorted(BACKUP.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    directories = [BACKUP, *(path for path in BACKUP.rglob('*') if path.is_dir())]
    for path in reversed(directories):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    copy_exact(manifest_path, artifacts / manifest_path.name)
    selected = [*(path for path in artifacts.glob('recovery-*') if path.is_file()), artifacts / manifest_path.name,
                *(ROOT / 'models').glob('recovery-*-verified.json'), ROOT / 'recover_weight_sync_inputs.py']
    evidence = {}
    for source in sorted(selected):
        relative = source.relative_to(ROOT)
        copy_exact(source, SETUP / relative)
        evidence[str(relative)] = {'bytes': source.stat().st_size, 'sha256': digest(source)}
    marker = {
        'status': 'ready', 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root': str(ROOT), 'backup_dir': str(BACKUP), 'durable_evidence_dir': str(SETUP),
        'verification': 'Full source and destination SHA256 checks completed by recover_weight_sync_inputs.py; this seal checks sizes, fsyncs durable files, and records file metadata for lightweight no-change admission.',
        'original_comparison_boundary': 'All 14 canonical NVFP4 file sizes and headers match retained original evidence; original full converted payload hashes were not retained.',
        'identity': manifest['identity'], 'inventory': inventory, 'inputs': inputs,
        'backup_manifest': {'path': str(manifest_path), 'sha256': digest(manifest_path)},
        'evidence': evidence,
    }
    target = artifacts / 'recovery-inputs-ready.json'
    with target.open('x') as stream:
        json.dump(marker, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    copy_exact(target, SETUP / 'artifacts' / target.name)
    for path in [*(path for path in SETUP.rglob('*') if path.is_dir()), SETUP, SETUP.parent]:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps({'marker': str(target), 'sha256': digest(target), 'inventory': inventory,
                      'files': len(inputs), 'bytes': sum(item['bytes'] for item in inputs.values())}))


if __name__ == '__main__':
    main()
