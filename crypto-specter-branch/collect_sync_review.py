#!/usr/bin/env python3
"""Collect Python sources from test_stand + crypto-specter-branch safely.

Run FROM test_stand/crypto-specter-branch:
    python3 ~/Downloads/collect_sync_review.py
Output: ~/sync_review_sources.zip
Never includes binary secrets, audio, logs, model weights, or .git data.
Review Python source code yourself before uploading if it contains embedded credentials.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import zipfile

SKIP_DIRS = {
    '.git', '.secrets', '.venv', 'venv', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', '.tox', '.idea', '.vscode', 'node_modules',
    'site-packages', 'dist', 'build', '.cache', 'recordings', 'records',
    'logs', 'tmp', 'temp', 'checkpoints', 'datasets', 'dataset', 'models',
}
EXTRA_NAMES = {
    'requirements.txt', 'requirements-dev.txt', 'requirements_test.txt',
    'pyproject.toml', 'setup.cfg', 'Pipfile', 'README.md', 'README_RU.md',
    'README_RU.txt', 'README.txt',
}
MAX_FILE_BYTES = 1_500_000


def main():
    ap = argparse.ArgumentParser(description='Collect main and branch Python source files (no secrets/audio)')
    ap.add_argument('--branch', default='.', help='Path to crypto-specter-branch (default: current directory)')
    ap.add_argument('--output', default=str(Path.home() / 'sync_review_sources.zip'))
    opts = ap.parse_args()
    branch = Path(opts.branch).expanduser().resolve()
    if branch.name != 'crypto-specter-branch':
        ap.error('Run from test_stand/crypto-specter-branch, or pass --branch PATH')
    main_dir = branch.parent
    if not (branch / 'experiment').is_dir() or not (main_dir / 'sync').is_dir():
        ap.error('Expected branch/experiment and parent test_stand/sync; check which project copy you are in')
    out = Path(opts.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines, found, skipped = [], [], []
    import_info = [f'Python executable: {sys.executable}', f'Python version: {sys.version}',
                   f'Platform: {platform.platform()}', f'MAIN_PROJECT: {main_dir}',
                   f'BRANCH: {branch}', f'ARCHIVE: {out}']
    sys.path.insert(0, str(branch))
    for mod in ('sync.v2_timepack', 'sync.v2', 'crypto.frame_port',
                'crypto.voiceproof_adaptor', 'crypto.spectral_timepack_adaptor',
                'experiment.syncVoiceProof', 'experiment.voiceProof20k',
                'experiment.spectralTimepack20k', 'experiment.telemostTest'):
        try:
            spec = importlib.util.find_spec(mod)
            import_info.append(f'{mod}: {getattr(spec, "origin", None) if spec else "NOT FOUND"}')
        except Exception as exc:
            import_info.append(f'{mod}: LOOKUP ERROR {type(exc).__name__}: {exc}')
    for label, source in [('main', main_dir), ('branch', branch)]:
        for dirpath, dirnames, filenames in os.walk(str(source), followlinks=False):
            where = Path(dirpath)
            dirnames[:] = sorted(n for n in dirnames if n not in SKIP_DIRS and
                                 not (label == 'main' and where == main_dir and n == branch.name))
            for filename in sorted(filenames):
                path = where / filename
                rel = path.relative_to(source)
                arc = f'{label}/{rel.as_posix()}'
                if path.is_symlink():
                    skipped.append(f'{arc}: symbolic link -> {os.readlink(str(path))}')
                    continue
                if not path.is_file() or not (path.suffix in ('.py', '.pyi') or filename in EXTRA_NAMES):
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped.append(f'{arc}: exceeds {MAX_FILE_BYTES} bytes')
                    continue
                found.append((path, arc))
    with zipfile.ZipFile(str(out), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for path, arc in found:
            payload = path.read_bytes()
            z.writestr(arc, payload)
            lines.append(f'{len(payload):>9} {hashlib.sha256(payload).hexdigest()}  {arc}')
        z.writestr('META/import_paths.txt', '\n'.join(import_info) + '\n')
        z.writestr('META/source_manifest.txt', '\n'.join(lines) + '\n')
        z.writestr('META/skipped_links_or_large_files.txt', '\n'.join(skipped) + '\n')
        z.writestr('META/README.txt', 'Python sources only. NO keys, WAV, npz, so, git, secrets or logs.\n'
                   'Check Python files for hard-coded tokens or passwords before sharing.\n')
    print('MAIN PROJECT:', main_dir)
    print('BRANCH:      ', branch)
    print('FILES:       ', len(found))
    print('ARCHIVE:     ', out, f'({out.stat().st_size} bytes)')
    print('\nACTUAL IMPORT LOOKUP:')
    for entry in import_info[6:]:
        print(' ',entry)
    print('\nBefore sharing: unzip -l',out)


if __name__ == '__main__':
    main()
