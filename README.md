# Extreme Weather–Transport–Economic Impact

Code supporting the manuscript:

**Coupled transport and supply-chain networks amplify economic losses from extreme weather**

## Overview

This repository provides workflows for analysing how extreme weather affects traffic flows and how transport disruptions propagate through regional production and supply-chain networks. The framework combines traffic forecasting and counterfactual analysis, regional origin–destination (OD) estimation, weather-intensity scenarios, and dynamic multiregional input–output (MRIO) simulation.

The research application covers New Zealand. The MRIO model represents 15 regions and 19 sectors, with daily simulation over 2019.

## Repository Structure

```text
extreme-weather-transport-economic-impact/
├── README.md
├── data/
├── minimal_example/
│   ├── dynamic_mrio_example.py
│   └── data/
│       └── MRIOTtest.xlsx
├── traffic_prediction/
├── od_estimation/
└── mrio_analysis/
```

| Directory | Description |
|---|---|
| `data/` | Shared traffic, weather, and network inputs |
| `minimal_example/` | Small-scale dynamic MRIO demonstration |
| `traffic_prediction/` | Traffic forecasting, sample preparation, training, and counterfactual analysis |
| `od_estimation/` | Regional OD disaggregation, traffic assignment, and iterative OD estimation |
| `mrio_analysis/` | Weather-intensity scenarios, transport disruptions, inventory preparation, and dynamic economic simulation |

## System Requirements

### Software and tested environment

The research modules require Python 3.10 or later. The minimal example was tested with:

| Component | Tested version |
|---|---|
| Operating system | Windows 11 25H2, 64-bit, build 26200.9711 |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 3.0.1 |
| Matplotlib | 3.11.2 |
| openpyxl | 3.1.5 |

This environment was used for the minimal example. Research-module dependency constraints and validation coverage are documented separately in each module. A common environment for the complete research workflow has not been validated.

### Hardware

The minimal example requires no specialized hardware. Measurements were taken on an AMD Ryzen 9 8945HX computer with 15.3 GiB of OS-visible physical memory. Installed graphics adapters are an NVIDIA GeForce RTX 5060 Laptop GPU and AMD Radeon(TM) 610M integrated graphics. No GPU was used for the minimal-example measurements. GPU acceleration is optional for traffic-model training.

## Installation

Clone the repository:

```bash
git clone https://github.com/transport-impact-code/extreme-weather-transport-economic-impact.git
cd extreme-weather-transport-economic-impact
```

### Minimal example

Create a virtual environment:

```bash
python -m venv .venv-demo
```

Activate it in Windows PowerShell:

```powershell
.\.venv-demo\Scripts\Activate.ps1
```

Or in macOS/Linux shells:

```bash
source .venv-demo/bin/activate
```

Install the versions used for the measured example:

```bash
python -m pip install numpy==2.3.5 pandas==3.0.1 matplotlib==3.11.2 openpyxl==3.1.5
```

Measured dependency installation time was approximately **67 seconds** in a fresh Python 3.12 virtual environment, downloading from public PyPI with pip caching disabled. This excludes virtual-environment creation. Download speed and caching affect installation time.

### Research modules

Use a separate environment for each module. From the repository root, run the command for the selected module in its environment:

```bash
# Traffic prediction environment
python -m pip install -r traffic_prediction/requirements.txt

# OD estimation environment
python -m pip install -r od_estimation/requirements.txt

# MRIO analysis environment
python -m pip install -r mrio_analysis/requirements.txt
```

## Demo

The bundled `minimal_example/data/MRIOTtest.xlsx` represents four regions and three sectors per region. The simulation runs for 40 time steps and includes capacity constraints, inventory consumption and replenishment, proportional allocation, and dynamic production adjustments.

Run from the repository root:

```bash
python minimal_example/dynamic_mrio_example.py
```

### Expected output

The script prints intermediate matrices and time steps, then displays a curve of total economic output. The simulated output array has shape **40 × 12**. The current script does not automatically save an output table or image.

### Measured runtime

Three runs took **6.23, 4.09, and 2.36 seconds**, with a median of approximately **4 seconds** on the computer described above. Timing used the noninteractive Matplotlib Agg backend and included process startup, imports, input loading, simulation, plot construction, and output checks. Interactive plot-viewing time was excluded. All three runs completed successfully and produced finite output arrays and 40-point output curves.

For a noninteractive run in Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python minimal_example/dynamic_mrio_example.py
```

For a noninteractive run in macOS/Linux shells:

```bash
MPLBACKEND=Agg python minimal_example/dynamic_mrio_example.py
```

The measurements cover the minimal example, not the full New Zealand workflows.

## Research Workflows

### Traffic prediction

Prepare traffic/weather data, generate samples, train the GCN–GRU model, and run prediction and counterfactual analysis. See [Traffic Prediction](traffic_prediction/README.md).

### OD estimation

Disaggregate regional OD demand to network access nodes, perform user-equilibrium assignment, and update regional demand using observed counts. See [OD Estimation](od_estimation/README.md).

### MRIO analysis

Construct weather-intensity scenarios, derive regional transport disruptions, embed them in annual production-capacity series, and simulate economic effects. H0 denotes official event duration; I1, I2, and I3 scale weather deviations from normal conditions by factors of 1, 2, and 3. See [MRIO Analysis](mrio_analysis/README.md).

## Using Your Own Data

Configure input/output paths in the relevant module. Align traffic and weather timestamps, match monitoring-site order to the adjacency matrix, and use consistent network identifiers and regional access nodes. Model weights must match the architecture and normalization parameters. MRIO matrices, capacity shocks, and inventories must use consistent region and sector ordering.

See [traffic inputs](traffic_prediction/docs/INPUT_DATA.md), [OD documentation](od_estimation/README.md), and [MRIO inputs](mrio_analysis/INPUT_DATA.md).

## Data Availability

The repository includes the small MRIO example and selected traffic, weather, network, and OD inputs. Full research workflows require additional compatible weights, normalization parameters, MRIO tables, and inventory data, as described in each module.

## Outputs and Validation

Outputs include traffic forecasts, counterfactual comparisons, OD matrices, capacity factors, daily economic output, and sectoral/regional losses. Research-module validation environments and completed checks are documented in:

- [Traffic prediction validation](traffic_prediction/validation.json).
- [OD estimation validation](od_estimation/validation.json).
- [MRIO analysis validation](mrio_analysis/VALIDATION.md).

The minimal example demonstrates economic propagation mechanisms. Reproducing the research workflows requires their additional inputs and configuration.
