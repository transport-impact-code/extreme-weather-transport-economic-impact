"""Verify bundled file hashes and compare a rerun with historical CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        path = root / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise AssertionError(f"Bundled file changed: {item['file']}")
    count = 0
    for scenario in ['real', 'cf']:
        relative = Path('2019-12-02') / scenario
        expected = root / 'reference_results' / relative
        actual = args.results / relative
        expected_names = sorted(p.name for p in expected.glob('*.csv'))
        actual_names = sorted(p.name for p in actual.glob('*.csv'))
        if not expected_names or expected_names != actual_names:
            raise AssertionError(f'Output file set differs: {relative}')
        for name in expected_names:
            pd.testing.assert_frame_equal(
                pd.read_csv(expected / name), pd.read_csv(actual / name),
                check_exact=False, rtol=1e-10, atol=1e-10,
            )
            count += 1
    print(f'PASS: {len(manifest)} bundled file hashes; {count} historical CSV comparisons.')


if __name__ == '__main__':
    main()
