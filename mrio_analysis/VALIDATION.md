# Validation record

The original project was read only; executable checks wrote results to the workspace. Detailed records are in `validation.json`, `docs/historical_comparisons.json`, and `docs/input_audit.json`.

|Check|Result|
|---|---|
|Sources|11 requested scripts and 3 supporting scripts fingerprinted; scientific source bytes unchanged by filename cleanup|
|S2|Executed from 12 existing prediction NPZ files; 12 daily-flow CSV comparisons passed|
|S3|Batch executed; 138 matching CSV comparisons passed|
|S5|Executed from 126 existing final OD tables; 126 completed-OD CSV comparisons passed|
|S6|63 intensity/date cases executed; 695 CSV comparisons passed|
|S7|Three intensities executed; 30 historical NPY comparisons passed|
|S8|Three intensities executed; 15 historical NPY comparisons passed|
|Inventory|Default and alternative branches executed; default 19-sector table matched its historical CSV|
|MRIO preparation|All 285 row/column labels checked; region-major inputs reordered for three intensities|
|S9|Three 365-period simulations executed without plotting; all 365×285 output arrays finite and nonnegative|

Historical comparisons cover numeric columns in 972 CSV files and complete arrays in 45 NPY files, with rtol=atol=1e-10. The largest absolute difference was 4.55e-13. Paths, H1/H0 labels and other nonnumeric metadata were excluded from numeric comparisons. Reordered MRIO maximum column and row balance errors were 2.18e-11 and 1.46e-11.

S9 `annual_loss_percent` values were 0.1179957311 (I1), 0.1654354587 (I2), and 0.1938712133 (I3). These are execution fingerprints for the available 21 event days, ETA=3.5 and the default inventory definition, not independently validated economic findings.

S1 neural inference, S4 network assignment/inversion, GUI plotting, and full dependency installation were not executed. Available validation used NumPy/Pandas; torch, networkx and matplotlib were absent. Official December 1 coverage and inventory mapping assumptions remain open.

Filename changes were checked against source SHA-256 fingerprints. The release test suite also checks ASCII Python filenames and the S1–S9 stage registry. Current test results are recorded in `validation.json`.
