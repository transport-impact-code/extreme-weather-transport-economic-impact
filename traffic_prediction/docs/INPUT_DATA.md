# Input Data Requirements

The original packaging task included six Python source files. Research datasets and trained model assets were not supplied with those files.

The paths below are configuration defaults, relative to the working directory. The repository stores the shared site list and adjacency matrix under the root-level `data/network/` directory; update the configuration paths accordingly.

| Stage | Required input | Required structure |
|---|---|---|
| `label` | `data/weather_flow_month/YYYYMM*.csv` | `DATETIME`; downstream processing also requires `SITEREF` and `FLOW`. Weather variables: `TEMP`, `PRCP`, `WDSP`, and `RH`; `MXSPD` is optional. The first six filename characters identify the month. Quantile thresholds are calculated across all matching files. |
| `samples` | Monthly `*_extreme_multihazard_event_v1.csv` files | `SITEREF`, `DATETIME`, `FLOW`, `TEMP`, `PRCP`, `WDSP`, `RH`, and `EXTREME_FLAG`. Missing auxiliary flag columns are filled with zero. |
| `samples` | `data/site_locations_130.csv` | A `SITEREF` column with one unique site per row. Values are converted to strings, stripped of whitespace, and padded to eight characters using `zfill(8)`. Row order defines the model's node order. |
| `train` / `analyze` | `data/adjacency_distance_130x130_exp.csv` | An `N × N` numeric adjacency matrix with an index column. Row and column order must match the site list. Defaults: `N = 130` and `ADJ_ALREADY_NORMALIZED = True`. |
| `train` | `artifacts/samples/YYYYMM_samples.npz` | `X_hist: (S, 24, N, 5)`, `X_future: (S, 12, N, 4)`, `Y: (S, 12, N, 1)`, and `EXTREME_FLAG: (S, 12, N, 1)`. |
| `analyze` | A selected monthly sample NPZ file | The same core arrays as training. Auxiliary flags are optional. Sample generation also saves `nodes: (N,)` and `times: (T,)`. |
| `analyze` | `stgnn_weather_multihazard_event_v1_24to12.pth` | A model `state_dict` compatible with the current `model.py`. The analysis script uses `torch.load`; load only trusted checkpoints. |
| `analyze` | `norm_stats_multihazard_event_v1.json` | Five-element `mean` and `std` arrays ordered as `FLOW`, `TEMP`, `PRCP`, `WDSP`, `RH`. These parameters must match the checkpoint's training normalization. |

