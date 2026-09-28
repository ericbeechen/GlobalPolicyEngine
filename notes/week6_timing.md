# Week 6: profile, then cache

The whole chain for one currency, as `scripts/regress.py` runs it: the market
path on every session, the nowcast on every business day, the model path, the
gap and z, and the backtest at all eight horizons. Wall-clock seconds on the
same MacBook (Apple silicon, Python 3.13, pandas 3.0.6), full sample, no profiler.

| | week 5 code | week 6 | sessions |
|---|---|---|---|
| USD | 136.4s | 43.8s | 4,110 (3,921 with a model path), through 2026-09-21 |
| GBP | 28.7s | 15.5s | 4,333 (4,063), through 2026-09-25 |
| both | 165s | 59s | |

Both week 6 runs are bit-identical to the week 5 references (`scripts/regress.py check`).

## What the profile said

Under cProfile, GBP ran about one third market panel, one third nowcast and one third model path. The same few calls sat under all three:

- **Whole-log sorts on every session.** `model.path.known` ran three times per session, and `cache.view` and `rate_in_force` once each. Each sorted the entire fixings or policy-rate log again. Now each log is sorted once (`cache.Presorted`). A read by publication date is then a prefix of the sorted log. Pandas' multi-key sort is stable (`np.lexsort`), so filtering the sorted log gives the rows in the order sorting the filtered log would.
- **Every vintage masked on every nowcast read.** `VintagePanel.as_of` masked the full frame, all series, for each series read. Now each series' rows are indexed once and read as numpy arrays (`series`, `published`).
- **Recomputing unchanged days.** The nowcast depends on its date only through which vintages are current, and on most days none of its inputs has a release. `VintagePanel.version` counts, per input series, the days its current rows changed. `nowcast.build` reuses the day before's row while that count stands. GBP's nowcast went from about 8s to 0.1s; this is most of USD's gain.
- The extractors computed each session's as-published rows twice, once in `published()` and once in `session()`. They now keep the last one.

What is left is spread thin: about 2ms a session in the market panel, and 1ms in the model path. No single call dominates, so it stops here. It is inside the one-to-two-minute target.
