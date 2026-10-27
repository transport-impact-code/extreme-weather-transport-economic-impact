import argparse
from pathlib import Path
from . import event, baseline


def main():
    p = argparse.ArgumentParser(description='Original NZ OD assignment and inversion; precomputed seed input only.')
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--mode', choices=['event', 'baseline'], default='event')
    p.add_argument('--dates', nargs='+', help='Default: original 2019-12-02 through 2019-12-08')
    p.add_argument('--scenarios', nargs='+', choices=['real', 'cf'])
    a = p.parse_args()
    base = a.data_dir
    m = event if a.mode == 'event' else baseline
    m.ACCESS_CSV = base / 'region15_access_nodes_main_component.csv'
    m.INITIAL_REGION_OD_CSV = base / 'region15_od_seed_from_gdp.csv'
    if any(a.output.resolve().is_relative_to(p.resolve()) for p in [a.data_dir, Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / 'reference_results']):
        p.error('Output cannot be inside input data, code, or reference results.')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be empty.')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'event':
        m.EDGE_INPUT_DIR = base / '201912'
        m.OUT_ROOT = a.output
        if a.dates:
            m.DATES = a.dates
        if a.scenarios:
            m.SCENARIOS = a.scenarios
    else:
        m.BASE_DIR = base
        m.EDGES_CSV = base / 'edge_input_daily_with_ferry.csv'
        m.ITER_DIR = a.output
    m.main()


if __name__ == '__main__':
    main()
