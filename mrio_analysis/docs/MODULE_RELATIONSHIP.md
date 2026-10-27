# Related modules

Place `scenario_analysis/` alongside `traffic_prediction/` and `od_estimation/`.

## Traffic prediction

The bundled `model.py` is byte-identical to the inspected traffic prediction model. The runner prefers a matching adjacent model file after a SHA-256 check. An explicit `shared_model_path` must match the bundled source. Samples, weights and normalization statistics must form a compatible set.

S1 adds three-intensity raw-weather interventions; the ordinary counterfactual analysis command is not a substitute for that calculation.

## OD estimation

Access nodes, initial regional OD and edge templates can be shared through configuration. Six common functions have identical ASTs: `ensure_required_columns`, `build_graph`, `aon_load`, `calc_tstt`, `expand_region_to_node_od`, and `global_scale_region_od`.

Other functions differ. In particular, S4 falls back to an unweighted mean when an OD group's weight sum is nonpositive, while the inspected OD module directly uses a weighted average. S4 is therefore retained. Matching default parameters alone does not establish full algorithm equivalence.

The existing OD module's December defaults start on December 2. Scenario source windows retain December 1; available data coverage is documented separately.
