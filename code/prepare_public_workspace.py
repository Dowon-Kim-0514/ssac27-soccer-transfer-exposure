"""Public filesystem setup only. No analysis code is imported or executed."""
import argparse
import csv
import datetime
import hashlib
import os
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
REQUIRED = [
    'kaggle/transfers.csv', 'kaggle/players.csv', 'kaggle/clubs.csv',
    'kaggle/games.csv', 'kaggle/competitions.csv', 'kaggle/appearances.csv',
    'reference/clean_data.csv', 'reference/fbref_combined.csv',
    'fbref_tables/standard_data.csv', 'fbref_tables/keepers_adv_data.csv',
]
DIRECTORIES = [
    'audit/reproduction/from_saved_clean/data/ML model',
    'audit/reproduction/audit_evidence', 'datasets/intermediate',
    'datasets/final', 'models', 'validation', 'positions', 'residuals',
    'figures', 'tables', 'reports', 'acquisition/downloaded_tables',
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def missing_inputs(repo=REPO):
    return [name for name in REQUIRED if not (repo/'data/inputs'/name).is_file()]


def initialize(repo=REPO):
    missing = missing_inputs(repo)
    if missing:
        raise ValueError('Missing private input files: ' + ', '.join(missing))
    work = repo/'data/work'
    # Refuse a partially populated workspace as well as an existing run.
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Work directory is not empty; use a fresh separate checkout.')
    for name in DIRECTORIES:
        (work/name).mkdir(parents=True, exist_ok=True)
    baseline = work/'audit/reproduction/from_saved_clean/data/clean_data.csv'
    shutil.copyfile(repo/'data/inputs/reference/clean_data.csv', baseline)
    (work/'METHODOLOGY_CHANGELOG.md').write_text(
        '# Local Reproduction Changelog\n\n'
        'Fresh public-I/O workspace; no historical analyses have been run by initialization.\n'
    )
    # Record actual authorized local inputs and code, not an empty placeholder
    # or the author's private absolute-path manifest.
    files = [repo/'data/inputs'/name for name in REQUIRED]
    files += sorted(p for p in (repo/'code').iterdir() if p.suffix in {'.py', '.R'})
    files += [baseline, work/'METHODOLOGY_CHANGELOG.md']
    records = []
    for i, path in enumerate(files, 1):
        relative = os.path.relpath(path, work)
        records.append(dict(
            result_id=f'ART{i:04d}', path=relative, relative_path=relative,
            artifact_type=path.suffix.lstrip('.'),
            methodology_version='PUBLIC_IO_INPUT_BASELINE',
            source_reference='Authorized local input or release code; no statistical evaluation',
            verification_status='FILE_HASH_ONLY', bytes=path.stat().st_size,
            sha256=sha256(path),
            modified_utc=datetime.datetime.fromtimestamp(
                path.stat().st_mtime, datetime.timezone.utc).isoformat(),
        ))
    with (work/'RESULTS_MANIFEST.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Read-only input existence check')
    group.add_argument('--initialize', action='store_true', help='Create fresh private work layout')
    args = parser.parse_args()
    missing = missing_inputs()
    if missing:
        print('NOT READY: authorized input files are missing:')
        for name in missing:
            print('  data/inputs/' + name)
        return 1
    if args.initialize:
        initialize()
        print('Initialized data/work. No model or statistical analysis was run.')
    else:
        print('Input files exist. This does not certify matching hashes, rights or numeric reproduction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
