# Open issues and validation boundaries

1. **December 1 coverage.** The official December window starts on 2019-12-01, but the original S1 time axis skips the first 24 hours (`Tin=24`). Available predictions and downstream data start on December 2. Filling the gap requires correctly aligned prior-month history and fresh validation; no dates or observations have been fabricated.
2. **Inventory definitions.** The default S9 branch uses AES Level 1 average opening/closing stocks and 1–90-day clipping. The alternative branch uses closing stocks/MRIO inputs and 1–30-day clipping. Both are retained as separate calculations.
3. **Sector mapping.** Owner-Occupied Property Operation has no AES match and receives median imputation (29.83339 days in the validated data). Combined MRIO sectors retain the original broader AES mappings. These approximations require methodological review.
4. **S1 and S4 execution.** Neural inference and network assignment/OD inversion were not rerun in the available environment. Validation used existing prediction NPZ files and 126 existing final OD tables. Source and parameter checks do not constitute end-to-end numerical validation.
5. **S9 reference results.** All three 365-period simulations ran with reordered inputs and produced finite, nonnegative outputs. No independent historical reference for the checked sector-major configuration was available.
6. **Plotting and installation.** GUI plotting and installation of the full dependency set were not tested. S9 was executed without plotting.
7. **OD module equivalence.** The S4 zero-weight-group fallback differs from the existing OD module. The scenario implementation is retained rather than replaced without equivalence evidence.
8. **Distribution.** Data and weights are not included. A source license has not been specified by the rights holder.
