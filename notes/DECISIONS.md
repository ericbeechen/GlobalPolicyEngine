# Decisions

One row per choice: what was decided, when, and why, with where it lives now.
This is the appendix of the research note, and the answer to "why is it like that?".

A new decision gets a row here in the same commit as the change. A change that
moves any frozen output (`scripts/regress.py check`) is a decision too: refreeze
only after its row is written, and say in it which numbers moved.

Consolidated in week 6 (2026-09-28) from the per-week notes, the comments in
`config/currencies.yml`, the module docstrings, the generated reports and the
README. `notes/macro.md` (last committed 2026-09-25) is folded in whole.
`notes/decisions.md` was never committed. What it said survives in the config
comments that cite it (by date: 2026-09-24, -27), but anything written there
and nowhere else still needs moving here.

Dates are when the choice was made, from the commits and the notes that cite
it. Weeks 1-5 ran from 2026-09-22 to 2026-09-27.

## Scope

| # | Decision | Date | Why |
|---|---|---|---|
| S1 | **Ship USD and GBP. EUR is out for now. A third currency waits on a data purchase, and it is your call.** Five were scoped. Two meet the bar on the data we have. EUR and AUD each fail it for a reason that money fixes, and code does not. | 2026-09-28 | The bar is a daily market-implied path over the next eight meetings, clean through 2020-22. **EUR** (`notes/spikes/EUR_2026-09-28.md`): MMSR publishes maintenance-period-dated OIS only as quarterly CCP aggregates (re-checked by hand: every MP-dated key is `FREQ=Q`, and the MP1 bucket has one value a year). CME €STR futures start 2022-10-31. The ECB's AAA curve short end carries the 2020-22 collateral distortion. ICE Euribor futures are the only daily route: Databento `IFLL.IMPACT` from 2018-12, a new paid dataset, plus a variable Euribor-€STR basis to model. **AUD** (`notes/spikes/AUD_2026-09-28.md`): the ASX 30-day IB future settles like ZQ and would drop into the futures backend (plus an OCR-target basis term of -8bp to 0 over 2020-25). Its history is not on Databento, and ASX sells it at A$2,760 per year (about A$28k from 2016, A$44k from 2010). The free RBA 1/3/6-month OIS stopped in December 2022 and reaches about four meetings. CAD was not scoped: the budget went to the two stronger candidates. |
| S2 | The time a third currency would have taken goes to the expression and evaluation layers (costs and carry, three portfolio variants, the note). | 2026-09-28 | Five done rigorously beats ten done badly, and the same holds at two. A third currency on contaminated or quarterly data would weaken every claim made about the first two. |
| S3 | If a data purchase is approved, EUR on Euribor futures comes before AUD. | 2026-09-28 | Euribor reuses the Databento pipeline and covers 2020-22 from 2018-12. The ECB's own data (€STR, DFR, MRO, HICP) is free and machine-readable for the model side. AUD also needs an ABS real-time vintage builder: ALFRED's Australian series are OECD copies from about 2013. |

## The market path

| # | Decision | Date | Why | Where |
|---|---|---|---|---|
| M1 | USD path from ZQ. Each contract month is one linear equation in the regime rates (calendar-day weights), all solved jointly by least squares. | 2026-09-22 | ZQ settles on the arithmetic calendar-month average of EFFR, so the extraction is closed-form. Joint least squares uses every contract, not a bootstrap that compounds errors down the strip. | `curves/policy_path.py` |
| M2 | Realized days substituted. On session t, fixings are known through t - 1, and weekends and holidays carry the last fixing. | 2026-09-24 | The NY Fed publishes t - 1's EFFR on the morning of t. The market settling on t did not know t's own. A Columbus Day session knows only through the Thursday. | `calendars.known_daily` |
| M3 | Pin the first regime to the last fixing when it spans fewer than 10 days (`path.min_regime_days: 10`). | 2026-09-24 | Below 10 days, one contract's price noise is amplified more than 3x. Over the 46 announcement days since 2021, the error in the implied step went from 5.8bp mean and 34bp worst to 0.5bp and 1.6bp. | config |
| M4 | Never drop the front contract (`path.min_forward_days: 0`). | 2026-09-24 | Once short regimes are pinned, dropping it gains nothing. Without pinning, it only moved the amplification into the next contract. | config |
| M5 | A session's settles are read as they stood at the end of the next business day (`market.final_after_bdays: 1`). The preliminary and the final settle are both kept. | 2026-09-24 | CME sends a preliminary settle at about 16:00 ET and the final that evening (Sunday for a Friday session). A later revision is not what the market traded on. Made a config key 2026-09-28. | `market._as_published` |
| M6 | Meeting calendar, point in time. An unscheduled meeting counts from its announcement. A cancelled one counts until it was cancelled. Scheduled meetings count on every date. | 2026-09-22 | The March 2020 emergency cuts were pillars only once announced. The meeting they replaced was priced until it was called off. fomc.csv records no publication date for the schedule: see lag L1. | `calendars.known_meetings` |
| M7 | Cross-check the ZQ path against SR1 (monthly), not SR3. The SOFR discount curve stays on SR3. First real SR1 session 2018-05-07. | 2026-09-25 | Over 2021-24, SR1 is closer to the realized basis in three of four years and smoother in every one. It averages like ZQ, so the basis is read month by month. SR1 lists only about 13 months, too few for a curve. CME's reference settles of 2018-04-20 to 05-04 were not a market. | `sofr.crosscheck`, `first_sessions` |
| M8 | GBP path from the Bank of England's fitted OIS spot curve: the forward between consecutive meetings, with the log discount factor linear between monthly nodes. Continuously compounded, on an Act/365 axis (`market.curve.year_days: 365`, a config key from 2026-09-28). | 2026-09-27 | There is no monthly-average future outside the US. The forward over a window is exact from two spot rates. Forwards from s·t match the Bank's published instantaneous forwards to 0.1bp; annual compounding would miss by up to 14bp. | `curves/forward.py` |
| M9 | GBP: always pin today-until-the-first-meeting to the rate in force (`market.pin_first_regime`). | 2026-09-27 | No meeting falls inside that window. Reading it off the curve smears the first step across a node interval. Against the SONIA that then printed: 0.84 to 0.52bp mean absolute error, and 3.29 to 0.09bp in 2022. | config |
| M10 | GBP rate in force = the last SONIA fixing, moved by any Bank Rate change since. | 2026-09-27 | A decision takes effect at noon on its announcement day, before any published fixing reflects it (25 decision days in the panel). | `market.rate_in_force` |
| M11 | GBP last regime: to the meeting after the eighth, or 45 days past the eighth where the calendar does not reach it (`path.tail_days: 45`). | 2026-09-27 | A whole window either way. A config key since 2026-09-28. | config |
| M12 | GBP validation: rebuild the Bank's own MPR conditioning path from the step path. | 2026-09-27 | It is the Bank's own benchmark. Over 29 reports from August 2019 to July 2026, the step path is within 1.0bp on average. The differences correlate -0.59 with the move priced inside the quarter, as a smooth spline against a step predicts. | `validation.conditioning` |

## The macro nowcast (from notes/macro.md, 2026-09-25)

| # | Decision | Date | Why |
|---|---|---|---|
| X1 | `as_of` = the end of day D: every value with `realtime_start <= D <= realtime_end`. | 2026-09-24 | Real time is the whole point of the nowcast. There is no "latest values" method; that is `as_of(today)`. |
| X2 | The activity gap is the unemployment gap, u - u*, with u* = CBO NROU from the same vintage, for the quarter of the unemployment month. Positive means slack. A missing u* raises and is never carried forward. | 2026-09-24 | Potential GDP is quarterly and rewritten in hindsight. The MPR's own term is 2(u_LR - u), with the opposite sign. |
| X3 | Inflation for the rule: 12-month core PCE, with PCE's latest one or two months bridged from core CPI. Fit PCE m/m = a + b x CPI m/m over the trailing 60 months in the same vintage. | 2026-09-25 | The MPR rules use four-quarter core PCE. A bridged month's error enters a 12-month rate once, but a 3-month annualized rate about four times. Real-time RMSE over 67 months: 0.093pp (bridge), 0.132 (CPI 1:1), 0.157 (last PCE carried). Windows of 36 and 120 months give 0.097 and 0.095; that is reported once, not tuned. |
| X4 | The October 2025 CPI hole is log-interpolated inside the proxy only. The gap uses November. | 2026-09-25 | Published as missing, so it stays NaN in the data. The two bridged months erred by +0.05 and -0.01pp. |
| X5 | Activity composite: quarter-to-date growth as a robust z (median/MAD over complete quarters from 1993 in the same vintage), equal-weighted. A diagnostic, never a rule input. | 2026-09-25 | Median/MAD means 2020Q2 needs no hand-picked exclusion. The GDPNow same-day correlation is +0.46 over 2021-26 and -0.25 since 2022. It is not reweighted to fit. |
| X6 | Real retail sales rebuilt as-of (RSAFS / CPIAUCSL), not FRED's RRSFS. | 2026-09-25 | RRSFS has no usable vintage history. |
| X7 | USD model path starts 2011-03-04. | 2026-09-25 | The first day every input has an ALFRED vintage (NROU 2011-02-02, average hourly earnings 2011-03-04). |
| X8 | "." artifacts in superseded vintages are kept as NaN. NROU is listed in `projections`, the only exemption from the published >= date guard. | 2026-09-25 | None touches a value the nowcast reads. CBO publishes ten years ahead. |
| X9 | GBP macro from the ONS's own revisions triangles (UNEM04, CLA03, PAYE RTI), with CPI (never revised) on its release days. Not ALFRED. | 2026-09-27 | FRED's OECD copies of UK series run 22-25 days late and stopped in 2025. |
| X10 | UK labour caveat, stated rather than smoothed. The LFS was suspended October 2023 to January 2024 (the last official vintage is held), and was "official statistics in development" February 2024 to August 2026. | 2026-09-27 | The LFS and claimant-count 12-month changes correlated +0.72 to 2022, and -0.16 since. The brief's footer carries it from the config tag. |
| X11 | GBP u* is a constant 4.5%. | 2026-09-27 | No vintaged UK natural rate is machine-readable back to 2010 (the OBR's are in forecast tables: a week 9 job). 4.5 is the Bank's "around 4½%" (February 2025 MPR). The traded z subtracts the gap's trailing mean, so a constant moves the bp level, not the trade. |

## The model path

| # | Decision | Date | Why |
|---|---|---|---|
| R1 | The rule is imposed, not estimated: the MPR's balanced-approach rule, R* = r* + π + 0.5(π - π*) + 2(u* - u), for both currencies. | 2026-09-27 | An estimated rule answers a different question and invites fitting the backtest. An estimated rule would be a different config block, not a rewrite. |
| R2 | Inertia 0.85 a quarter, per meeting as 0.85^(1/meetings per quarter). USD 2 meetings a quarter. GBP 3 until 2015 and 2 from 2016. | 2026-09-27 | The MPR's inertial rule is quarterly. The MPC met monthly until 2015. |
| R3 | Hold-flat conditioning: π, u - u* and r* stay at today's values at every meeting ahead. | 2026-09-27 | It answers "if nothing changes, where does the rule take policy?" without forecasting the macro state. That is also what would make a reading wrong, and the brief says so. |
| R4 | The ELB is a floor on R*, applied before the inertial step. The unfloored notional is kept. USD 0.125 (midpoint of 0-0.25). GBP 0.5 until 2016-08-04, then 0.1. | 2026-09-27 | The model does not go below the central bank's own floor, and what the rule asks for there stays on record (977 USD sessions). The MPC described 0.5, then "close to, but a little above, zero", and went to 0.1 and never below. The floor is dated in config: a zero floor would be flatly wrong for EUR (DFR -0.50). |
| R5 | USD r* = the SEP median longer-run funds rate minus 2%, step-filled, with Taylor (1993)'s 2% before the first dot (2012-01-25). | 2026-09-27 | It models the committee's reaction function, so its own belief is the relevant number. Each SEP is dated and never revised, so the series is real-time with no vintages. FRED rounds to 0.1, so r* is within 5bp. HLW is the week 9 robustness check. |
| R6 | GBP r* = a constant -1.6%. | 2026-09-27 | No longer-run rate is published, and the long end of the OIS curve would make the model inherit market pricing. -1.59 is the 2009-08 to 2026-08 average of Bank Rate minus CPI (median -1.40). A hindsight number: it moves the bp gap, not the z. |
| R7 | Operating spread: the overnight rate minus the policy rate, as the median of the last 20 fixings, held flat. GBP estimates it only on fixings the same side of the SONIA reform (2018-04-23). | 2026-09-27 | A median, because EFFR printed 5-12bp low on most month ends in 2015-17. SONIA went from brokered to transaction-based. |
| R8 | Policy rate identity, from config: the USD target-range midpoint, GBP Bank Rate. `rule.policy_rate` may be a dated schedule of series. | 2026-09-27; schedule 2026-09-28 | The ECB's operative rate is the DFR only since excess liquidity made it so (about 2014). Before then it is the MRO: a change point inside the sample. |
| R9 | `rule.inflation_target` may be a dated schedule. | 2026-09-28 | The ECB went from "below, but close to, 2%" to a symmetric 2% in July 2021, a reaction-function change point. Nothing uses it yet. |

## The signal and the backtest

| # | Decision | Date | Why |
|---|---|---|---|
| G1 | Gap = market minus model, in bp, per meeting horizon. Both bp and z are kept. | 2026-09-27 | The z is what gets traded, and the bp is what gets talked about. |
| G2 | z against each horizon's own trailing two calendar years (`730D`), not counting today, after 250 sessions of history. | 2026-09-27 | Pricing errors are regime-dependent, and a longer window mixes ELB years with cycles whose gaps are not comparable. Today is left out so a z uses only what was known before it. |
| G3 | sd floored at 5bp. | 2026-09-27 | At the ELB the gap barely moves, and a 1-2bp sd would turn a tick of noise into a large z. |
| G4 | The first cross-country signal is GBP - USD, gap minus gap, on sessions both have, z-scored the same way. | 2026-09-27 | Positive means GBP prices more tightening against its rule than USD does against its own. Most of the level is the two r* choices; the z is what moves. |
| B1 | The crude backtest trades k = 4, chosen before any P&L was run. The other horizons are a sensitivity, not a menu. | 2026-09-27 | USD Sharpe 0.02 (2012-26: it loses 2014-20 and makes it back 2021-25). GBP 0.75, 90% of it from 2022-23. One regime each: no evidence yet. |
| B2 | Receive when z > 0, DV01 fixed, one session's execution lag. P&L is credited to the meeting held into the session, not the k-th on each day. | 2026-09-27 | Measuring the k-th meeting's rate on each day would book the roll from one meeting to the next as P&L. |
| B3 | No transaction costs yet. | 2026-09-27 | Week 8. The config tag puts this in the brief's footer. |

## Publication lags and approximations (every one)

| # | Series | Rule | Approximation |
|---|---|---|---|
| L1 | FOMC and MPC scheduled dates | Known on every date. | Neither calendar records when a date was published. The Fed publishes about two years ahead. The September 2022 MPC meeting is listed on the 22nd, where it moved to. |
| L2 | EFFR, SOFR | The next Fed business day. | Checked against ALFRED's first-seen dates (`scripts/pull_fred.py`). |
| L3 | Target range, SEP median | The same day (lag 0). | The range in force on d is announced before d starts. An SEP is dated on its release day. |
| L4 | CME settles | Final by the end of the next business day. | Sunday for a Friday session. |
| L5 | SONIA; Bank Rate; the Bank's curve | The next London business day; the same day; by noon the next business day. | |
| L6 | ONS CPI before February 2016, LFS before April 2016 | Dated the 26th of the month. | Before the ONS release calendar starts: up to about 10 days late, never early. |
| L7 | Core PCE | The latest one or two months bridged from core CPI. | Real-time RMSE 0.075pp m/m. |
| L8 | NROU | Vintages from 2011-02-02 only. | So the USD model path starts 2011-03-04. |
| L9 | PAYE RTI | Vintages from December 2019 only. | Not used before then. |
| L10 | FRED current-vintage revisions | Stamped with the day we retrieved them. | FRED reports a revised value under its original publication date. Ours is the earliest we can vouch for. |
| L11 | r* (USD) | FRED rounds the SEP median to one decimal. | Within 5bp. |

## Week 6: engineering (2026-09-28)

| # | Decision | Why |
|---|---|---|
| E1 | Snapshot, then refactor. `scripts/regress.py` freezes every stage (sessions, meetings, macro, model, paths, signal, backtest at every horizon) and checks it bit for bit, with no tolerance. There are two tiers. The fixtures tier is committed in `tests/data/reference/` and checked by pytest. The full tier is the whole sample, in `data/reference/`. The USD full reference was frozen on the untouched week-5 code, from the cache on the Windows machine. The GBP one was frozen on the week-5 code from a fresh pull. | Without it, a moved number after a refactor cannot be told apart from an improvement. Proved by breaking things on purpose. A 1e-12 change in a coefficient, or 1e-9pp on one fixing, is caught. So are an early SEP, early fixings, late market data and early vintages. It also caught one bug during the performance work: a `max(initial=NaT)` that made every nowcast's `published` NaT. |
| E2 | Every enabled currency block is validated against `config.SCHEMA` on load, and fails with every problem listed by its dotted path. It also checks that every series the chain reads is cached by some entry under `sources`, and that every daily series has a publication lag. A block with `enabled: false` is a draft and is not checked. | A third currency is an hour of config, not an afternoon of KeyErrors. |
| E3 | Sources are named in config (`sources.daily`, `curves`, `futures`; `macro.source`) and resolved in `sources/registry.py`. | Adding a provider is a module and a registry line. `update_data.py` is not edited. |
| E4 | No function defaults to a currency's calendar. `bday` is always passed. | A forgotten argument silently used the Fed's calendar. |
| E5 | `tests/test_no_currency_branches.py` fails if a shared module names a currency, its rates, futures, committee or calendar in code. Docstrings and comments may explain with examples. Backends (a source, a currency's own report or check) are listed with the reason. | The acceptance grep, kept true from now on. |
| E6 | The look-ahead suite runs for every currency with fixtures (`tests/data/fixtures.yml`), through the production extractors on the fixtures laid out as a cache. | Week 5 checked two hand-written pipelines. Now a new currency is a fixtures block. |
| E7 | Fixings pass two filters: `cache.view`, then `calendars.known_daily`. A one-day slip in either one alone moves no output, so the pipeline test cannot see it. Each filter has its own unit test. | Found by planting the leak: defence in depth hides a broken layer, so each layer is pinned directly. |
| E8 | Performance: sort each vintage log once, not on every session (`cache.Presorted`). Index vintages by series, and read them as arrays. Reuse a nowcast row on days none of its inputs changed (`VintagePanel.version`). Both full-sample references stay bit-identical. | The profile showed repeated full-log sorts and masks, not anything structural. The full chain went from 136s to 44s for USD and from 29s to 15s for GBP: both currencies in under a minute (`notes/week6_timing.md`). |
| E9 | The USD brief said "unemployment for the three months to <month>". UNRATE is monthly, so the wording is now `report.labels.unemployment_period`. The model report headed GBP's CPI column `core_pce`; it is now `inflation`. The USD coverage report names the holiday behind a CME-closed Friday ("Independence Day: CME closed, Fed open"), not "Saturday holiday": the same 13 days, classified the same way. | Wording bugs found in the inventory. Report text changes; no frozen number moves. |
| E10 | `notes/DECISIONS.md` is committed. `notes/decisions.md` is no longer gitignored. | Git's ignore matching is case-insensitive on macOS and Windows, so the old line ignored this file too. Un-ignored, a local `notes/decisions.md` stops a pull instead of being silently overwritten: git clobbers ignored files. |
| E11 | A check nowcasts through the last session it built, never through the end date read back from `meta.json`. | `pd.date_range` takes its resolution from its end point: a parsed date gave `as_of` as `[us]`, where the freeze had `[ns]`. The harness flagged it, and it is a harness artifact, not a change to the pipeline. |
| E12 | A fresh vintage pull may re-print a cached value: if it is within 1e-12 relative of the cached one, that is not a revision, and the cached bits are kept (`cache.SAME_VALUE`). A larger change to a cached vintage still refuses to write. | ALFRED now prints 41 old NROU vintages at full float precision (`5.693902493` came back as `5.6939024929999995`). The old exact-equality guard stopped every USD update at NROU, and the six series after it would never have refreshed. Keeping the cached bits keeps outputs frozen on the cache bit-identical. |
| E13 | Report figures fall back from Segoe UI to Arial, then DejaVu Sans. | Segoe UI exists only on Windows. On macOS the old fallback, DejaVu Sans, is wide enough to push the brief and the one-pager past one page. Arial sets text at about Segoe's width. Windows output is unchanged, since it never reaches the fallback. |
