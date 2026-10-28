"""Build a reproducible MAF submission ZIP from an explicit file allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = [
    'LICENSE', 'README.md', 'requirements.txt', 'requirements-scout.txt',
    'run.py', 'validate.py', 'build_submission.py',
    'algorithms/__init__.py', 'algorithms/preprocessing/__init__.py',
    'algorithms/preprocessing/MinorityClassWeighting.py',
    'data/synthetic.csv', 'data/generate_synthetic.py',
    'results/scout_reported.json', 'results/maf_benchmarks.json',
    'experiments/scout/train_edu_gbm.py', 'experiments/scout/test_edu_gbm.py',
    'experiments/scout/source_manifest.json',
]


def build():
    entries = {}
    for name in FILES:
        path = ROOT / name
        if path.is_symlink():
            raise ValueError(f'Symlinks are not allowed: {path}')
        entries[name] = path.read_bytes()
    snapshots = json.loads(entries['experiments/scout/source_manifest.json'])
    for name, source in snapshots.items():
        if hashlib.sha256(entries['experiments/scout/' + name]).hexdigest() != source['sha256']:
            raise ValueError(f'SCOUT source snapshot changed: {name}')
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    entries['MANIFEST.sha256.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode()
    directory = ROOT / 'dist'
    directory.mkdir(exist_ok=True)
    archive = directory / 'ALINJob_MAF_submission.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo('ALINJob_MAF_submission/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
    with zipfile.ZipFile(archive) as output:
        if output.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
    print(f'{archive}: {len(entries)} files; SHA256 {hashlib.sha256(archive.read_bytes()).hexdigest()}')
    return archive


if __name__ == '__main__':
    build()
