# Scenario Analysis

Evaluate the effects of extreme-weather intensity on traffic flows, regional origin–destination (OD) demand, and economic output. Traffic-derived disruptions feed a dynamic multiregional input–output (MRIO) model covering 15 regions, 19 sectors, and 365 days in 2019.

## Scenarios

H0 is the official event duration. I1, I2, and I3 scale weather deviations from normal conditions by factors of 1, 2, and 3. All intensities use the same event windows.

|Event|H0 window, inclusive|
|---|---|
|201903|2019-03-23 to 2019-03-27|
|201908|2019-08-10 to 2019-08-14|
|201911|2019-11-19 to 2019-11-22|
|201912|2019-12-01 to 2019-12-08|

## Setup

Use Python 3.10 or later. From this directory:

```console
python -m pip install -r requirements.txt
```

Prepare traffic/weather samples, trained weights, normalization statistics, network and OD inputs, an MRIO table, and AES 2019 financial data. Data and weights are supplied separately; see [INPUT_DATA.md](INPUT_DATA.md).

Copy `config.example.json` to `config.json` and set `data_root`. The default scenario directory is `data_root/第五阶段情景分析new/I_H0/`. Use `paths_by_stage` to specify a different input or output layout. `null` retains a default path; candidate-path entries use arrays. Relative paths resolve from the working directory.

## Run

Run the traffic and annual disruption stages in order:

```console
python run.py S1 --config config.json --no-plots
python run.py S2 --config config.json
python run.py S3 --config config.json
python run.py S4 --config config.json
python run.py S5 --config config.json
python run.py S6 --config config.json
python run.py S7 --config config.json
python run.py S8 --config config.json
```

For S9, prepare the sectoral inventory table and sector-major MRIO inputs if they are not already available:

```console
python run.py S9_inventory --config config.json
python run.py S9_prepare --config config.json
```

Then run the economic simulation:

```console
python run.py S9 --config config.json --no-plots
```

`S9_prepare` converts the S8 MRIO matrices to sector-major ordering. It is a data preparation step, needed only when the required checked inputs have not been prepared. `--no-plots` disables figures in S1 and S9; omit it to enable plotting.

|Script|Purpose|
|---|---|
|`S1.py`|Traffic predictions for three intensities and a normal-weather counterfactual|
|`S2.py`|Hourly and daily node flows within event windows|
|`S3.py`|Road-edge flow inputs|
|`S4.py`|Daily regional OD estimation|
|`S5.py`|Regional OD diagonal completion|
|`S6.py`|Transport capacity factors and disruptions|
|`S7.py`|Daily OD and regional disruption series|
|`S8.py`|Annual MRIO production-capacity series|
|`S9.py`|Dynamic economic simulation and loss summaries|
|`S9_inventory.py`|AES-based sectoral inventory days used by S9|
|`S9_prepare.py`|Sector-major MRIO input preparation|

`S9_inventory_legacy.py` and `S9_inventory_variants.py` provide an alternative inventory calculation and its sensitivity variants. Their outputs do not replace the default S9 inventory automatically.

## Outputs and related modules

Outputs include traffic predictions, node and edge flows, OD matrices, regional disruptions, annual capacity series, daily economic output, sectoral and regional losses, event summaries, and model diagnostics.

Use this directory alongside `traffic_prediction/` and `od_estimation/`. Existing model assets and network/OD data can be shared through configuration. See [module interfaces](docs/MODULE_RELATIONSHIP.md).

## Checks

```console
python -m unittest discover -s tests -v
```

Available prediction and intermediate data omit 2019-12-01 and cover 21 event days; the source retains the H0 windows above. See [validation](VALIDATION.md) for executed checks and unverified stages, and [open issues](UNRESOLVED.md) for remaining data and methodological questions.
