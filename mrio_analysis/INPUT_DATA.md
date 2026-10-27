# Input data and stage interfaces

Set `data_root` in `config.json`. The default scenario root is `data_root/第五阶段情景分析new/I_H0/`. Override paths under the corresponding stage key in `paths_by_stage` when using another layout. Candidate-path keys accept arrays; other path keys accept strings. A value of `null` keeps the default. Relative paths resolve from the command working directory.

|Stage|Inputs|Outputs|
|---|---|---|
|S1|Monthly samples, adjacency, weights, normalization statistics, model|`H0/I*/{month}_national_multihazard_event_results.npz`|
|S2|S1 NPZ: `pred_real`, `pred_cf`, `true`, and prediction timestamps|Hourly/daily node flows|
|S3|Daily node flows, node-edge mapping, edge observations and template|Daily real/counterfactual edge inputs|
|S4|Edge inputs, regional access nodes and initial regional OD|`region_od_final.csv` for each date and flow case|
|S5|S4 regional OD and region order|OD with completed diagonals|
|S6|Real/counterfactual completed OD|Regional and OD capacity factors|
|S7|Daily S6 factors and matrices|Regional factors, OD arrays, region and date indices|
|S8|S7 outputs and raw MRIO CSV|Raw MRIO arrays and annual capacity shocks|
|S9_inventory|AES 2019 CSV and S8 sector names|Sectoral inventory days for S9|
|S9_prepare|S8 MRIO arrays and the same raw MRIO CSV|`MRIO模型_sector_major_checked`|
|S9|Checked sector-major arrays and sectoral inventory days|Daily dynamic output, loss summaries and diagnostics|
|S9_inventory_legacy|AES closing stocks and MRIO intermediate inputs|Alternative inventory table, clipped to 1–30 days|
|S9_inventory_variants|Alternative inventory table|Low/middle/high tables|

## Prediction inputs

S1 uses 130 nodes, 24 historical hours and a 12-hour forecast horizon. Samples include `X_hist`, `X_future`, `Y`, and `EXTREME_FLAG`. Raw weather channels are TEMP, PRCP, WDSP, and RH.

Within flagged event observations, the counterfactual replaces raw weather with normal conditions. I1/I2/I3 use `normal + k * (observed - normal)`, with k=1/2/3. Normalized future-weather inputs remain fixed. Existing physical clipping is retained.

## MRIO preparation

S8 outputs Z (285×285), F (285×15), VA/X (1×285), and capacity factors (19×15×365). S9 expects sector-major activity indices, `sector * 15 + region`.

`S9_prepare.py` reorders both Z axes, the F activity axis, and VA/X activity axes when needed. The capacity array is already sector×region×time. Previously prepared, compatible checked inputs can be reused directly.

## Inventory inputs

The default S9 table comes from `S9_inventory.py`: average opening/closing stocks divided by operating inputs, multiplied by 365, with H19 preferred and H17+H18 used as fallback; values are clipped to 1–90 days. The original sector mapping and median imputation are retained.

The alternative `S9_inventory_legacy` branch uses closing stocks divided by MRIO intermediate inputs and clips to 1–30 days. `S9_inventory_variants` reads that branch's output and applies low/middle/high multipliers of 0.5/1/2 with its original bounds. It does not change the default S9 input.

## Coverage and fingerprints

The official event windows span 22 days. Existing predictions start on 2019-12-02 for the December event, leaving 2019-12-01 uncovered and 21 available event days overall. Date windows and time-axis formulas are unchanged.

Data and weights are not bundled. `data_manifest.json` records project-relative paths and SHA-256 fingerprints of inputs used in validation. Original source names are retained in provenance metadata for traceability; all published Python filenames are ASCII.
