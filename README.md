# Extreme Weather–Transport–Economic Impact

Code supporting the manuscript:

**Coupled transport and supply-chain networks amplify economic losses from extreme weather**

## Overview

This repository provides computational workflows for analysing how extreme weather affects traffic flows and how transport disruptions propagate through regional production and supply-chain networks.

The framework combines:

- Spatiotemporal traffic forecasting and weather counterfactual analysis.
- Regional origin–destination (OD) estimation and traffic assignment.
- Weather-intensity scenario analysis.
- Dynamic multiregional input–output (MRIO) simulation.

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
| `traffic_prediction/` | Traffic forecasting, sample preparation, model training, and counterfactual analysis |
| `od_estimation/` | Regional OD disaggregation, traffic assignment, and iterative OD estimation |
| `mrio_analysis/` | Weather-intensity scenarios, transport disruptions, inventory preparation, and dynamic economic simulation |

## System Requirements

### Software

The research modules require Python 3.10 or later. Dependencies are specified separately for each module:

| Component | Dependencies |
|---|---|
| Minimal example | NumPy, pandas, Matplotlib, openpyxl |
| Traffic prediction | NumPy, pandas, Matplotlib, PyTorch |
| OD estimation | NumPy, pandas, NetworkX |
| MRIO analysis | NumPy, pandas, NetworkX, PyTorch, Matplotlib |

Dependency constraints are provided in each research module’s `requirements.txt`.

### Tested Environments

- Operating system: **[operating system and version]**
- Minimal-example environment: **[Python, NumPy, pandas, Matplotlib, and openpyxl versions]**

Recorded research-module validation environments include:

| Module | Python | NumPy | pandas |
|---|---|---|---|
| OD estimation | 3.12.14 | 2.3.5 | 3.0.1 |
| MRIO analysis | 3.12.14 | 2.3.5 | 3.0.1 |

These versions describe module-specific validation. They do not represent a single validated environment for the complete workflow. Additional validation details are provided in the corresponding modules.

### Hardware

The minimal MRIO example uses a standard CPU and requires no specialized hardware. GPU acceleration is optional for traffic-model training.

The computer used for the installation and demo timing measurements had:

- CPU: **[CPU model]**
- RAM: **[RAM capacity]**
- GPU, if used: **[GPU model or “Not used”]**

## Installation

Download the repository or clone it:

```bash
git clone https://github.com/transport-impact-code/extreme-weather-transport-economic-impact.git
cd extreme-weather-transport-economic-impact
```

### Minimal Example

Create and activate a virtual environment:

```bash
python -m venv .venv-demo
```

On Windows PowerShell:

```powershell
.\.venv-demo\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv-demo/bin/activate
```

Install the example dependencies:

```bash
python -m pip install numpy pandas matplotlib openpyxl
```

Typical installation time on the computer described above: **[installation time]**, using **[network or package-cache conditions]**.

### Research Modules

Use a separate virtual environment for each research module. From the repository root, run the installation command corresponding to the selected module:

```bash
# Traffic prediction environment
python -m pip install -r traffic_prediction/requirements.txt
```

```bash
# OD estimation environment
python -m pip install -r od_estimation/requirements.txt
```

```bash
# MRIO analysis environment
python -m pip install -r mrio_analysis/requirements.txt
```

Module-specific configuration and execution instructions are provided in each module’s README.

## Demo

### Input Data

The minimal example uses the bundled file:

```text
minimal_example/data/MRIOTtest.xlsx
```

It represents a system with four regions and three sectors per region.

### Run

From the repository root:

```bash
python minimal_example/dynamic_mrio_example.py
```

The simulation runs for 40 time steps and includes:

- Production-capacity constraints.
- Inventory consumption and replenishment.
- Proportional order allocation.
- Dynamic production adjustments.

At each time step, sectoral output is determined by inventory availability, production capacity, and incoming orders.

### Expected Output

The script prints intermediate matrices and simulation steps to the terminal and displays a plot of total economic output over time.

The current example displays the figure interactively; it does not automatically save an output table or image.

### Expected Runtime

The demo takes approximately **[demo runtime]** on the computer described above, excluding dependency installation and time spent viewing the plot.

## Research Workflows

### Traffic Prediction

Prepare traffic and weather data, generate forecasting samples, train the GCN–GRU model, and perform prediction and counterfactual analysis.

See [Traffic Prediction](traffic_prediction/README.md) for input formats, configuration, and commands.

### OD Estimation

Disaggregate regional OD demand to network access nodes, perform user-equilibrium traffic assignment, and update regional OD demand using observed traffic counts.

See [OD Estimation](od_estimation/README.md) for the supplied example, input data, and reference-result checks.

### MRIO Analysis

Construct three weather-intensity scenarios, derive regional transport disruptions, embed the disruptions in an annual production-capacity series, and simulate economic effects through the MRIO model.

The scenarios use the official event duration H0. I1, I2, and I3 scale weather deviations from normal conditions by factors of 1, 2, and 3, respectively.

See [MRIO Analysis](mrio_analysis/README.md) for the S1–S9 workflow, inventory inputs, matrix preparation, and outputs.

## Using Your Own Data

Configure input and output paths within the relevant module. Ensure that:

- Traffic observations, weather variables, and timestamps are aligned.
- Monitoring-site order matches the adjacency matrix.
- Network identifiers and regional access nodes match the OD inputs.
- Model weights match the model architecture and normalization parameters.
- MRIO matrices, capacity shocks, and inventory tables use consistent region and sector ordering.

Detailed file formats and configuration instructions are provided in:

- [Traffic prediction inputs](traffic_prediction/docs/INPUT_DATA.md).
- [OD estimation documentation](od_estimation/README.md).
- [MRIO analysis inputs](mrio_analysis/INPUT_DATA.md).

## Data Availability

The repository includes the small-scale MRIO demonstration dataset and selected traffic, weather, network, and OD inputs.

Additional inputs are required for the full research workflows, including compatible trained model weights, normalization parameters, MRIO tables, and inventory data. Input requirements and available datasets are described in the corresponding modules.

## Outputs and Validation

Outputs include traffic forecasts, counterfactual comparisons, estimated OD matrices, transport-capacity factors, daily economic output, and sectoral and regional loss summaries.

Validation environments, reference-result comparisons, and the scope of completed checks are documented separately:

- [Traffic prediction validation](traffic_prediction/validation.json).
- [OD estimation validation](od_estimation/validation.json).
- [MRIO analysis validation](mrio_analysis/VALIDATION.md).

The minimal example demonstrates the economic propagation mechanism. Reproducing the research workflows requires the additional inputs and configuration described in the module documentation.
