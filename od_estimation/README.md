# New Zealand Extreme Weather Research: OD Traffic Assignment and Inversion

This repository provides reproducible code and input data for regional origin–destination (OD) demand disaggregation, traffic assignment, and iterative OD inversion in a New Zealand extreme weather research project.

The implementation preserves the computational functions, formulas, constants, scenario order, and output conventions of the original research scripts. File paths are configurable so that the code can run independently of the original project directory.

## Scope

The workflow includes:

1. Reading a precomputed regional OD seed matrix.
2. Disaggregating regional OD demand into node-level OD demand using access-node weights.
3. Performing traffic assignment using shortest paths, the Bureau of Public Roads (BPR) travel-time function, and the Method of Successive Averages (MSA).
4. Updating regional OD demand using observed and simulated directional edge flows.
5. Applying global demand scaling and the original stopping criteria.
6. Processing real and counterfactual scenarios for the December 2019 event window.

The initial OD seed is supplied as `data/region15_od_seed_from_gdp.csv`. **Code for generating the initial OD seed is not included.**

Weather and traffic prediction, counterfactual edge-input generation, diagonal completion, multiregional input–output (MRIO) modelling, and dynamic economic loss simulation are outside the scope of this repository. The required edge inputs are provided as CSV files.

## Requirements

Python 3.10 or later is required.

Install the dependencies from the repository root:

```console
python -m pip install -r requirements.txt
```

The dependency versions in `requirements.txt` match the environment used for the reported validation.

## Quick Start

Run the validated single-day example for both real and counterfactual scenarios:

```console
python -m nz_od --dates 2019-12-02 --scenarios real cf
```

Results are written to:

```text
results/
└── 2019-12-02/
    ├── real/
    └── cf/
```

Compare the generated results with the bundled historical reference results:

```console
python verify.py
```

A successful check reports that all 71 bundled file hashes and all 54 historical CSV comparisons have passed.

Execution may take several minutes because the original shortest-path calculations are retained.

## Full Event Window

To process all seven days from 2 to 8 December 2019:

```console
python -m nz_od --output results_full
```

The default workflow processes each date in chronological order, running `real` first and `cf` second, for a total of 14 cases.

To run the baseline OD inversion workflow:

```console
python -m nz_od --mode baseline --output results_baseline
```

Use an empty output directory for each run. The command refuses to overwrite a non-empty output directory.

Input files are read from the bundled `data/` directory by default. An alternative input directory can be supplied using `--data-dir`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── provenance.json
├── data_manifest.json
├── validation.json
├── verify.py
├── nz_od/
│   ├── __init__.py
│   ├── __main__.py
│   ├── baseline.py
│   └── event.py
├── data/
│   ├── region15_od_seed_from_gdp.csv
│   ├── region15_access_nodes_main_component.csv
│   ├── edge_input_daily_with_ferry.csv
│   └── 201912/
│       ├── edge_input_real_2019-12-02.csv
│       ├── edge_input_cf_2019-12-02.csv
│       └── ...
└── reference_results/
    └── 2019-12-02/
        ├── real/
        └── cf/
```

- `nz_od/baseline.py`: baseline traffic assignment and OD inversion functions extracted from the original second-stage S9 script.
- `nz_od/event.py`: event-window traffic assignment and OD inversion functions extracted from the original Step3 script configured for December 2019.
- `nz_od/__main__.py`: command-line interface and portable path configuration.
- `data/`: three shared input CSV files and 14 daily scenario-specific edge-input CSV files.
- `reference_results/`: 54 historical output CSV files for the validated single-day example.
- `verify.py`: bundled-file integrity checks and numerical comparisons against the reference results.
- `provenance.json`: original source-script identifiers and SHA-256 hashes.
- `data_manifest.json`: bundled input and reference-result file paths, sizes, and SHA-256 hashes.
- `validation.json`: recorded function-consistency and numerical validation results.

## Computational Settings

The original settings are retained:

| Parameter | Value |
|---|---:|
| BPR alpha | 0.15 |
| BPR beta | 4.0 |
| Minimum directional capacity | 1e-6 |
| MSA step size at iteration `it` | `1 / it` |
| Maximum UE iterations | 50 |
| UE relative-gap tolerance | 1e-4 |
| Maximum outer OD iterations | 5 |
| OD update exponent | 0.20 |
| Minimum adjustment factor | 0.50 |
| Maximum adjustment factor | 1.50 |
| Numerical stabilizer, EPS | 1e-6 |
| Global scaling at each outer iteration | Enabled |
| Factor-mean stopping tolerance | 0.01 |
| OD-total-ratio stopping tolerance | 0.01 |

The implementation also retains the original handling of missing values, unreachable OD pairs, diagonal entries, and operation order.

The original `DiGraph` behaviour for parallel edges is preserved: later entries for the same directed node pair overwrite earlier entries.

## Outputs

Each case produces the original per-iteration outputs:

- Node-level OD demand.
- Directional edge flows and travel times.
- UE iteration logs.
- Regional adjustment factors.
- Updated regional OD demand.
- Final regional OD demand.
- An outer-iteration summary.

The event workflow also produces `batch_od_inversion_summary.csv`.

Original output filenames, columns, and UTF-8 BOM encoding are retained. Path fields in the batch summary depend on the selected input and output locations.

As in the original workflow, the edge-flow output from the last outer iteration corresponds to the OD demand before that iteration’s OD update.

## Validation

The portable package was rerun using only the bundled input data for:

- `2019-12-02 / real`
- `2019-12-02 / cf`

All 54 case-level CSV outputs matched the historical reference results within a relative tolerance of `1e-10` and an absolute tolerance of `1e-10`. All 71 bundled input and reference-result files passed SHA-256 integrity checks.

The extracted function bodies in the baseline and event modules were also checked against the original source scripts and found to be identical at the abstract syntax tree level.

The remaining six event dates and the baseline workflow have not undergone a complete historical-output rerun. Their inclusion does not imply that all cases have been numerically validated.

## Data and Reproducibility Boundary

The precomputed initial OD seed and scenario-specific edge inputs are treated as fixed inputs to this workflow. Reproducing their upstream generation is outside the scope of this repository.

This repository reproduces the OD assignment and inversion stage, from the supplied seed and edge inputs to the updated regional OD outputs.
