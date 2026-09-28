# Week 6 inventory: every currency-conditional outside `config/`

Day 3 of week 6, listed before anything was moved. Each row says where the value
lived, what it is, and where it went. "Kept" rows are backends: a module the
config *names* (a data source, an extractor, a currency's own validation report).
Naming a backend is how a currency differs without the module branching on it.
Adding a currency writes a new backend only when its data needs a new source.

Searched with `grep -rnE 'USD|GBP|EFFR|SONIA|SOFR|FOMC|MPC|ZQ|SR[13]|US_BDAY|UK_BDAY|365|360|"20..-..-.."'`
over `src/` and `scripts/`, then read module by module.

## Leaks: currency behaviour hardcoded in shared code

| # | Where | What was hardcoded | Moved to |
|---|---|---|---|
| 1 | `market.py` `settles_on`, `solve_session`, `rate_in_force`, `solve_curve`; `calendars.known_daily` | `bday=US_BDAY` defaults: a caller that forgot the calendar silently got the Fed's | No default. Every caller passes `BDAYS[cfg["calendar"]]` |
| 2 | `market._as_published` | market data is final "by the end of the next business day" for every market | `market.final_after_bdays` (USD 1: CME's final settle; GBP 1: the Bank's curve by noon) |
| 3 | `market.solve_curve` | `tail_days=45`: the last regime's close when the calendar has no later meeting | `path.tail_days` |
| 4 | `curves/forward.py` | `YEAR_DAYS = 365.0` ("SONIA accrues Act/365") | `market.curve.year_days` |
| 5 | `market.FuturesStripExtractor`, `scripts/run_implied_path.py` | the cache namespace `"databento"` for the policy futures | `market.futures: {source, series}` (was `policy_futures`) |
| 6 | `report/coverage.py` `checks` | `US_BDAY`; "Fed business days", "CME closed, Fed open", "no ZQ settles"; `USFederalHolidayCalendar` for the exchange's Friday closures | bday passed in; words from `report.labels`; exchange closures from `market.exchange_calendar` (`calendars.EXCHANGES`) |
| 7 | `report/charts.py` | `LABELS` default = USD's words | none: `report.labels` always passed |
| 8 | `report/figures.py` | `"ZQ futures"`, `"EFFR"` defaults; the annotated session `2021-12-01` | `market.label`, `market.overnight_label` required; `report.annotate_from` |
| 9 | `report/model.py` | column `core_pce` (GBP's is CPI); "Taylor's 2%", "SEP dot" | column `inflation`; `rule.rstar.label` |
| 10 | `report/brief.py` | "a new SEP moved r*"; "unemployment for the three months to" (true of the LFS, wrong for UNRATE, which is monthly) | `report.labels.rstar_update`, `report.labels.unemployment_period` |
| 11 | `scripts/build_brief.py` | horizons `[1, 4, 8]` in the printout | `brief.yml` `horizons` |
| 12 | `scripts/run_implied_path.py` | `CCY = "USD"`; ZQ-only solve | `--ccy`; `market.extractor(ccy)` |
| 13 | `scripts/run_nowcast.py` | `CCY = "USD"`; core PCE / bridge / activity printout | `--ccy`; prints the fields the nowcast returned |
| 14 | `scripts/check_mpr.py` | truth file `tests/data/boe/conditioning_paths.csv`, example reports, `--ccy GBP` default | `validation.conditioning: {truth, examples}`; runs for every currency that has one |
| 15 | `scripts/update_data.py` | one config key per source (`fred_series`, `fred_lags`, `boe_series`, `boe_lags`, `boe_curves`, `futures_roots`) and the macro-source table in the script; a lag of 1 when a series was not listed | `sources:` block (`sources/__init__.py` `DAILY`, `CURVES`, `FUTURES`, `MACRO` registries); every daily series lists its lag, and the schema refuses one that does not |
| 16 | `scripts/build_panel.py`, `curves/basis.py` | `"databento"` for the cross-check root; `settles_on` on the Fed default | `sofr.crosscheck` names the root; `market.futures.source`; bday passed |
| 17 | `macro/nowcast.py` | `nowcast(as_of, ccy="USD", ...)` | `ccy` required unless `panel` and `spec` are both given |
| 18 | `model/reaction.py`, `model/rstar.py` | `inflation_target` a constant (the ECB's changed in July 2021) | may be a dated schedule, like `elb` and `meetings_per_quarter` |
| 19 | `model/path.py` `inputs` | the policy rate is one pair of series for the whole sample (the ECB's operative rate is the MRO before ~2014, the DFR after) | `rule.policy_rate` (was `target_range`): one `{source, series}` or a dated schedule of them |
| 20 | `report/nowcast.py` | `EXCLUDE_QUARTER = "2025-01-01"` (GDPNow's gold-import quarter) | `macro.activity.benchmark_exclude` |
| 21 | `report/labour.py` | series codes `MGSX`, `CLAIMANTS`, `PAYE`; the LFS suspension window | `macro.labour: {lfs, claimants, paye, suspended}` |
| 22 | `config/currencies.yml` itself | USD had no `spread_breaks`; lags implicit | explicit `spread_breaks: []`; every lag listed |

## Kept: backends a config names

| Where | Why it stays |
|---|---|
| `sources/fred.py`, `boe.py`, `ons.py`, `rates.py`, `archive.py` | a data source: the network or the archive for one provider. It uses its provider's calendar (FRED publishes on Fed days, the Bank on London days) |
| `sources/pull_fomc.py`, `pull_mpc.py` | regenerate one committed meetings file each. The emergency-meeting dates in them are facts about that calendar |
| `calendars.py` `FedHolidayCalendar`, `UKHolidayCalendar` | a holiday calendar, named by `calendar:`. A third currency adds one (TARGET would be five rules) |
| `curves/basis.py`, `helpers.py`, `build.py`, `report/crosscheck.py` | the SOFR cross-check, a USD feature that runs where the config has a `sofr` block. `ql.Sofr()` is the SOFR index by definition |
| `curves/futures.py` Act/360 | what SR3 settles to, by CME's rule (`contract_shapes`) |
| `report/nowcast.py`, `report/labour.py` | each currency's nowcast report, named by `macro.report` |
| `report/conditioning.py` | the check against the Bank's own MPR conditioning paths, named by `validation.conditioning` |
| `report/onepager.py` | the week 4 one-pager, USD prose throughout. Superseded by the brief; `build_model.py --onepager` now refuses a currency whose config does not set `report.onepager: true` |
| `report/vintages.py` `STATS_FROM`, `SINCE` | report windows, the same for every currency |

## Found on the way (not currency leaks, fixed or logged)

- `brief.changed` said "unemployment for the three months to <month>" for USD too. UNRATE is a monthly rate. Fixed by #10, and logged in DECISIONS.md, because the USD brief's wording changes.
- `report/model.py` headed GBP's inflation column `core_pce`. Fixed by #9; the GBP report's table header changes.
