# Traffic Prediction

GCN–GRU traffic forecasting and weather counterfactual analysis. The model uses 24 hours of historical traffic and weather, together with 12 hours of future weather, to predict traffic flow for the next 12 hours.

## Installation

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare the following inputs and update their paths in `configs/default.json`:

- Monthly traffic and weather CSV files
- A monitoring-site list defining the node order
- An adjacency matrix matching that node order

For analysis with an existing trained model, provide its checkpoint and matching normalization JSON file.

See [Input Data](docs/INPUT_DATA.md) for the required columns and array formats.

## Usage

Run the following commands from the project directory:

```bash
# Label weather extremes and events
python -m traffic_prediction label --config configs/default.json

# Generate samples
python -m traffic_prediction samples --config configs/default.json

# Train the model
python -m traffic_prediction train --config configs/default.json

# Run predictions and counterfactual analysis
python -m traffic_prediction analyze --config configs/default.json
```

If samples are already available, start from training. If a checkpoint and its matching normalization parameters are available, run analysis directly.

To analyze another month, update both `MONTH` and `MONTH_NPZ` in the configuration. Relative paths are resolved from the directory where the commands are run.

## Project Structure

```text
traffic_prediction/
├── label_data.py        # Weather and event labeling
├── build_samples.py     # Sample generation
├── dataset_train.py     # Data loading and normalization
├── model.py             # GCN–GRU forecasting model
├── train.py             # Model training
├── plot_all_multi.py    # Prediction and counterfactual analysis
└── __main__.py          # Command-line interface

configs/                 # Input and output configuration
scripts/                 # Pipeline scripts
tests/                   # Smoke tests
docs/                    # Data requirements and technical notes
```

## Model and Counterfactual Analysis

The model combines two graph convolutional layers, a GRU encoder, and an autoregressive GRUCell decoder. This version includes an explicit weather penalty and a directional counterfactual training loss.

Two predictions are generated using the same trained model:

- **Real-weather prediction:** uses the original weather inputs.
- **Counterfactual prediction:** replaces selected raw weather values with reference weather conditions.

The normalized future-weather input, `x_future`, remains unchanged. Only selected values in `x_future_raw` are replaced.

Labeling and analysis use different event windows. Both original definitions are preserved and documented in [Code Notes](docs/CODE_AUDIT.md).

## Outputs

Outputs are saved under `artifacts/` by default:

- Sample NPZ files
- Trained model weights and normalization parameters
- Training loss curves
- Real-weather and counterfactual predictions
- Impact summaries and comparison plots


