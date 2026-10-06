# Decisions

One row per choice: what was decided, when, why, and where it lives. This is the appendix of the research note and the answer to "why is it like that?".

A new decision gets a row in the same commit as the change. A change that moves a frozen output (`scripts/regress.py check`) is a decision too: write its row, saying which numbers moved, before refreezing.

Dates are when the choice was made. Results rows quote the run on data through 2026-09-29 (a few earlier ones say "to 2026-09-21"). The committed reports are from the run through 2026-10-01, which moves some figures slightly (the book's net Sharpe from -0.62 to -0.60) and changes no reading. `reports/numbers.md` lists every number the note quotes, linked to the report line that prints it.

This file was condensed on 2026-10-04. The unabridged version, with every intermediate count and review note, is in git history before that date.

## Scope

- **S1** (2026-09-28). **Ship USD and GBP. EUR is out for now; a third currency waits on a data purchase.** *Why:* the bar is a daily market-implied path over the next eight meetings, clean through 2020-22. Five currencies were scoped and two meet it. EUR fails it: MMSR publishes maintenance-period OIS only as quarterly aggregates, CME €STR futures start 2022-10-31, the ECB's AAA curve carries the 2020-22 collateral distortion, and the only daily route, ICE Euribor futures, is a new paid Databento dataset (`notes/spikes/EUR_2026-09-28.md`). AUD fails it: the ASX 30-day IB future would drop into the futures backend, but its history is not on Databento and ASX sells it for about A$28k-44k, and the free RBA OIS stopped in December 2022 (`notes/spikes/AUD_2026-09-28.md`). CAD was not scoped.
- **S2** (2026-09-28). The time a third currency would have taken goes to expression and evaluation (costs, carry, the portfolio variants, the note). *Why:* a third currency on contaminated or quarterly data would weaken every claim made about the first two.
- **S3** (2026-09-28). If a data purchase is approved, EUR on Euribor futures comes before AUD. *Why:* Euribor reuses the Databento pipeline and the ECB's model-side data is free. AUD also needs an ABS real-time vintage builder.

## The market path

- **M1** (2026-09-22). USD path from ZQ: each contract month is one linear equation in the regime rates (calendar-day weights), all solved jointly by least squares. *Why:* ZQ settles on the calendar-month average of EFFR, so the extraction is closed-form, and a joint fit uses every contract instead of compounding errors down a bootstrap. `curves/policy_path.py`
- **M2** (2026-09-24). Realized days are substituted: on session t, fixings are known through t - 1, and weekends and holidays carry the last fixing. *Why:* the NY Fed publishes t - 1's EFFR on the morning of t. `calendars.known_daily`
- **M3** (2026-09-24). Pin the first regime to the last fixing when it spans fewer than 10 days (`path.min_regime_days`). *Why:* below 10 days one contract's price noise is amplified more than 3x. Over the 46 announcement days since 2021, the error in the implied step went from 5.8bp mean and 34bp worst to 0.5bp and 1.6bp.
- **M4** (2026-09-24). Never drop the front contract (`path.min_forward_days: 0`). *Why:* once short regimes are pinned, dropping it gains nothing; without pinning it only moved the amplification into the next contract.
- **M5** (2026-09-24). A session's settles are read as they stood at the end of the next business day (`market.final_after_bdays: 1`); preliminary and final are both kept. *Why:* CME sends a preliminary around 16:00 ET and the final that evening (Sunday for a Friday). `market._as_published`
- **M6** (2026-09-22). The meeting calendar is point in time: an unscheduled meeting counts from its announcement, a cancelled one until it was cancelled. *Why:* the March 2020 emergency cuts were pillars only once announced. `calendars.known_meetings`
- **M7** (2026-09-25). Cross-check the ZQ path against SR1, not SR3; the SOFR discount curve stays on SR3. First real SR1 session 2018-05-07. *Why:* SR1 averages like ZQ, and over 2021-24 it was closer to the realized basis in three of four years and smoother in all four. It lists only about 13 months, too few for a curve. `sofr.crosscheck`
- **M8** (2026-09-27). GBP path from the Bank of England's fitted OIS spot curve: the forward between consecutive meetings, log discount factor linear between monthly nodes, continuously compounded, Act/365. *Why:* there is no monthly-average future outside the US, and the forward over a window is exact from two spot rates. `curves/forward.py`
- **M9** (2026-09-27). GBP: always pin today-until-the-first-meeting to the rate in force (`market.pin_first_regime`). *Why:* reading it off the curve smears the first step; pinning cut the error against SONIA from 0.84bp to 0.52bp.
- **M10** (2026-09-27). GBP rate in force is the last SONIA fixing, moved by any Bank Rate change since. *Why:* a decision takes effect at noon on its announcement day, before any fixing reflects it. `market.rate_in_force`
- **M11** (2026-09-27). GBP last regime runs to the meeting after the eighth, or 45 days past the eighth if the calendar ends (`path.tail_days`).
- **M12** (2026-09-27). Validate GBP by rebuilding the Bank's own MPR conditioning path from the step path. *Why:* it is the Bank's own benchmark. Over 29 reports the step path is within 1.0bp on average, and the differences correlate -0.59 with the move priced inside the quarter, as a smooth spline against a step predicts. `validation.conditioning`

## The macro nowcast (from notes/macro.md, 2026-09-25)

- **X1** (2026-09-24). `as_of` is the end of day D: every value with `realtime_start <= D <= realtime_end`. There is no "latest values" method. *Why:* real time is the point of the nowcast.
- **X2** (2026-09-24). The activity gap is the unemployment gap, u - u*, with u* the CBO NROU from the same vintage. A missing u* raises. *Why:* potential GDP is quarterly and rewritten in hindsight.
- **X3** (2026-09-25). Inflation is 12-month core PCE, with PCE's latest one or two months bridged from core CPI (PCE m/m on CPI m/m over the trailing 60 months, same vintage). *Why:* the MPR rules use four-quarter core PCE, and the bridge beat CPI 1:1 and carrying the last PCE in real time.
- **X4** (2026-09-25). The October 2025 CPI hole is log-interpolated inside the bridge only; the data keeps it missing.
- **X5** (2026-09-25). An activity composite (robust z of quarter-to-date growth) is a diagnostic, never a rule input.
- **X6** (2026-09-25). Real retail sales are rebuilt as-of (RSAFS / CPIAUCSL). *Why:* RRSFS has no usable vintage history.
- **X7** (2026-09-25). The USD model path starts 2011-03-04. *Why:* the first day every input has an ALFRED vintage.
- **X8** (2026-09-25). "." artifacts in superseded vintages are kept as NaN; NROU is the only exemption from the published-date guard (CBO publishes ten years ahead).
- **X9** (2026-09-27). GBP macro comes from the ONS's own revisions triangles, with CPI (never revised) on its release days. *Why:* FRED's OECD copies run late and stopped in 2025.
- **X10** (2026-09-27). The UK labour caveat is stated, not smoothed: the LFS was suspended October 2023 to January 2024 (last official vintage held) and was "in development" until August 2026. *Why:* LFS and claimant-count changes correlated +0.72 to 2022 and -0.16 since.
- **X11** (2026-09-27). GBP u* is a constant 4.5%, the Bank's "around 4½%". *Why:* no vintaged UK natural rate is machine-readable back to 2010 (see V8).

## The model path

- **R1** (2026-09-27). The rule is imposed, not estimated: the MPR's balanced-approach rule, R* = r* + π + 0.5(π - π*) + 2(u* - u), for both currencies. *Why:* an estimated rule answers a different question and invites fitting the backtest.
- **R2** (2026-09-27). Inertia 0.85 a quarter, per meeting as 0.85^(1/meetings per quarter): USD 2, GBP 3 until 2015 and 2 from 2016.
- **R3** (2026-09-27). Hold-flat conditioning: π, u - u* and r* stay at today's values at every meeting ahead. *Why:* it answers "if nothing changes, where does the rule take policy?" without forecasting the macro state.
- **R4** (2026-09-27). The ELB is a floor on R*, applied before the inertial step; the unfloored notional is kept. USD 0.125, GBP 0.5 until 2016-08-04 then 0.1. *Why:* the model does not go below the central bank's own floor, and what the rule asks for there stays on record.
- **R5** (2026-09-27). USD r* is the SEP median longer-run funds rate minus 2%, with Taylor's 2% before the first dot (2012-01-25). *Why:* it models the committee's reaction function, so the committee's own belief is the number; each SEP is dated and never revised, so it is real time without vintages.
- **R6** (2026-09-27). GBP r* is a constant -1.6%, the 2009-2026 average of Bank Rate minus CPI. *Why:* no longer-run rate is published, and the long end of the OIS curve would make the model inherit market pricing. A hindsight number.
- **R7** (2026-09-27). Operating spread (overnight rate minus policy rate) is the median of the last 20 fixings, held flat; GBP uses only fixings on the same side of the 2018 SONIA reform. *Why:* EFFR printed low on most month ends in 2015-17.
- **R8** (2026-09-27; schedule 2026-09-28). The policy rate comes from config (USD target-range midpoint, GBP Bank Rate) and may be a dated schedule. *Why:* the ECB's operative rate changes inside the sample.
- **R9** (2026-09-28). `rule.inflation_target` may be a dated schedule (for the ECB's 2021 change). Nothing uses it yet.
- **R10** (2026-09-28). No operating-spread break at the 2016-03-01 EFFR method change. *Why:* the spread moved by one basis point, the smallest step rounding allows. No number moves.

## The signal and the backtest

- **G1** (2026-09-27). Gap = market minus model, in bp, per meeting; both bp and z are kept. *Why:* the z is traded, the bp is talked about.
- **G2** (2026-09-27). z against each horizon's own trailing two calendar years, not counting today, after 250 sessions. *Why:* pricing errors are regime-dependent, and a longer window mixes ELB years with cycles.
- **G3** (2026-09-27). The sd is floored at 5bp. *Why:* at the ELB a 1-2bp sd would turn a tick of noise into a large z.
- **G4** (2026-09-27). The cross-country signal is GBP - USD, gap minus gap, z-scored the same way.
- **B1** (2026-09-27). The crude backtest trades the fourth meeting, chosen before any P&L was run; other horizons are a sensitivity, not a menu.
- **B2** (2026-09-27). Receive when z > 0, DV01 fixed, one session's lag; P&L is credited to the meeting held, so the roll between meetings is not booked as P&L.
- **B3** (2026-09-27). No transaction costs in the crude backtest (superseded by K1-K13).

## Publication lags and approximations

| # | Series | Lag | Note |
| --- | --- | --- | --- |
| L1 | FOMC and MPC scheduled dates | Known on every date | Neither calendar records when a date was published. |
| L2 | EFFR, SOFR | Next Fed business day | Checked against ALFRED's first-seen dates. |
| L3 | Target range, SEP median | Same day | |
| L4 | CME settles | Final by end of next business day | Sunday for a Friday session. |
| L5 | SONIA; Bank Rate; the Bank's curve | Next London day; same day; noon next day | |
| L6 | ONS CPI before Feb 2016, LFS before Apr 2016 | Dated the 26th | Before the ONS release calendar: up to ~10 days late, never early. |
| L7 | Core PCE | Latest 1-2 months bridged from CPI | Real-time RMSE 0.075pp m/m. |
| L8 | NROU | Vintages from 2011-02-02 | So the USD model path starts 2011-03-04. |
| L9 | PAYE RTI | Vintages from Dec 2019 | Not used before then. |
| L10 | FRED current-vintage revisions | Stamped the day retrieved | The earliest date we can vouch for. |
| L11 | USD r* | FRED rounds to 0.1 | Within 5bp. |
| L12 | Treasury CMT par yields | Next Fed business day | No print when the bond market is shut but the Fed or CME is open; a leg carries its last print. A fitted curve, not traded bonds. |
| L13 | Moody's Baa, Aaa (BAA10Y, AAA10Y) | Next Fed business day | Seasoned bonds: spread changes stand in for excess returns, and stale quotes can lag. |
| L14 | ICE BofA IG and HY OAS | Next Fed business day | FRED keeps three years (from 2023-09-29). A cross-check only. |
| L15 | DEXUSUK | Same day | The noon New York rate, a market price; not the London close. |
| L16 | The Bank's gilt spot curve | Noon next London day | The 0.5y point is often missing; interpolating it moves par yields by under 0.4bp. A fitted curve. |
| L17 | HLW r*, NY Fed real-time vintages | Quarter end + 65 days, or the release day if later | 65 days is the shortest lag with no vintage seen early. No vintages 2020Q3-2022Q3: the 2020Q2 one stands through the gap. |

## Engineering (2026-09-28)

- **E1**. Snapshot, then refactor: `scripts/regress.py` freezes every stage and checks it bit for bit. Fixtures tier committed in `tests/data/reference/`; full tier in `data/reference/`. *Why:* without it a moved number cannot be told from an improvement. It catches a 1e-12 change in a coefficient, and it caught one real bug.
- **E2**. Every enabled currency block is validated against `config.SCHEMA` on load, listing every problem by path. *Why:* a third currency should be an hour of config, not an afternoon of KeyErrors.
- **E3**. Sources are named in config and resolved in `sources/registry.py`. *Why:* a new provider is a module and a registry line.
- **E4**. No function defaults to a currency's calendar; `bday` is always passed. *Why:* a forgotten argument silently used the Fed's.
- **E5**. `tests/test_no_currency_branches.py` fails if a shared module names a currency in code.
- **E6**. The look-ahead suite runs for every currency with fixtures, through the production extractors.
- **E7**. Fixings pass two filters (`cache.view`, `calendars.known_daily`), and each has its own unit test. *Why:* defence in depth hides a broken layer.
- **E8**. Performance work (presorted vintage logs, array reads, reused nowcast rows) with both references bit-identical. The full chain fell from 136s to 44s (USD) and 29s to 15s (GBP).
- **E9**. Wording fixes in the brief and reports. No frozen number moves.
- **E10**. `notes/DECISIONS.md` is committed and `notes/decisions.md` is no longer gitignored. *Why:* ignore matching is case-insensitive on macOS and Windows.
- **E11**. A regress check nowcasts through the last session it built, not the date read back from `meta.json` (a timestamp-resolution artifact).
- **E12**. A re-printed vintage within 1e-12 relative of the cached value is not a revision (`cache.SAME_VALUE`). *Why:* ALFRED began printing old NROU vintages at full precision, which stopped every USD update.
- **E13**. Report figures fall back from Segoe UI to Arial, then DejaVu Sans. *Why:* DejaVu is wide enough to push one-pagers past a page on macOS.

## Engineering: live market data (2026-09-30)

- **E14**. Futures settles after the batch archive's last day come from Databento's historical API, with the same dataset, schema and symbols, into the same vintage log. On when `DATABENTO_API_KEY` is set; `--archive-only` turns it off. A live day is refetched until Databento calls it `available`. A day Databento calls degraded is refetched on every update (about a cent a day). *Why:* the archive ended 2026-09-21, leaving USD seven sessions behind GBP. The API returns the batch records exactly, and no frozen number moves. `sources/rates.py`
- **E15**. Every API request is priced first (`metadata.get_cost`) and refused past `rates.LIVE_BUDGET_USD`, set at $1. *Why:* a day of ZQ, SR1 and SR3 is about $0.01, so only a mistake reaches the budget.

## Expression and carry (2026-09-28)

- **C1**. Outright instruments: USD, the ZQ month that settles on the fourth meeting's regime; GBP, the OIS forward over [E4, E5) off the Bank's curve, read with the path's own helper. `strategy/instruments.py`
- **C2**. ZQ DV01 comes from the contract spec, never typed ($41.67 a contract); CME's published values are data, and a test fails if the number appears in code. *Why:* a typed number goes stale silently.
- **C3**. Interpolation is the simplest each data source allows: ZQ months linear in maturity, forwards linear in -log P, par yields linear between tenors.
- **C4**. A leg's mark on day d is the print dated d as it stood after its publication lag; a missing print carries the last one. Never `cache.read(as_of=t)`. *Why:* that returns t - 1's print and books the move between decision and execution as P&L. `instruments.prints`
- **C5**. Instrument ids are unique in time (a ZQ contract is its expiration month). *Why:* ZQF1 is both January 2011 and January 2021.
- **C6**. Timing: a position decided at t's close executes one session later; a cross sleeve at the slower lag of its pair. Reproduces the crude backtest to 7e-15bp.
- **C7**. Carry: a par leg receives its coupon and pays the overnight rate; futures and forwards are unfunded. GBP legs convert at DEXUSUK, and the build refuses a spot outside [1.0, 2.5]. *Why:* funding at overnight stands in for repo; the range catches an inverted quote.
- **C8**. Convexity is not in the linear P&L; the full revaluation measures it (about 2.6-2.7bp a year per unit DV01 against a receiver of 2s10s).
- **C9**. Three checks guard the P&L: the carry + roll + rate identity; an independent full revaluation inside a Taylor bound (no session outside it); and convergence tests per sleeve. The build refuses to write the report if one fails. *Why:* the identity alone cannot prove the signs.
- **C10**. "Right and bleeding" is receiving on an inverted curve or paying on an upward-sloping one. The sign is easy to get backwards, so twelve worked cases pin each one. `tests/test_carry.py`
- **C11**. The breakeven's edge is the de-meaned gap (|z| × sd), not the raw bp gap, and the expected quarter is E_h = φ × edge + (1 - φ) × CR_h. *Why:* the raw gap carries constant offsets (r*, u*, the spread) that z removes.
- **C12**. Carry and roll ahead (CR_h) is over 91 days for the instrument held, curve frozen, signed for the side.
- **C13**. The closure φ_h is an expanding-window, no-intercept slope, clipped to [0, 1], from 250 pairs. A signal does not pay for its bleed when E_h < 0. *Why:* real time; a full-sample φ would let 2022 decide whether a 2015 signal paid.
- **C14**. The traded slope is orthogonalised against the level over a trailing 730 days; the raw slope is a robustness row. *Why:* unorthogonalised, the slope's z correlates 0.85 (USD) and 0.94 (GBP) with the level's.
- **C15**. Both 2s10s are government curves (Treasury CMT, gilt par from the Bank's curve); the cross sleeve is 2y against 2y. *Why:* the OIS curve reaches 25 years only from 2016, and a cross built from the outright legs would make the covariance singular.
- **C16**. The expression report trades the linear rule (s = z) with no costs; the expression layer takes any series of positions.
- **C17**. The breakeven horizon is a quarter, 91 days, for carry, closure and outcome alike. *Why:* two meetings, about how long a trade is held.
- **C18**. A signal is a session with |z| ≥ 1, scored on its next quarter. Of 3,938 signal sessions right on the rate, 19% lost once carry and roll were counted; 53% bled. The reading: whether bleeding is a reason to stand aside turns on 2022 (bleeding signals earned more over the full sample, less without 2022), so the carry filter stays a diagnostic. `report/expression.py`
- **C19**. The brief's limitations footer reads only the tags of the blocks the brief shows. *Why:* room for the trades table; the full list goes to the generated limitations.
- **C20**. The brief's trades table: per sleeve, z, side, instrument, edge, CR_h, E_h and the verdict. Checked on 769 Friday briefs: all fit one page.

## Costs (2026-09-29)

Numbers are from `reports/costs.md`: hysteresis (1, 0), flat at the ELB, vol-scaled, configured costs.

- **K1**. Costs per leg, one way = half the round trip. ZQ: two back-month ticks plus the fee, 1.048bp a round trip, computed from the contract spec. Treasuries 0.5bp, OIS forward and gilts 1bp, all assumed. *Why:* USD is observable with the tick as the floor; GBP is assumed, which is what the sensitivity curve is for.
- **K2**. No bid-offer is measured: the archive holds settlements, not quotes, so ZQ's cost is its tick. A quote-based estimate would need a data purchase.
- **K3**. Charged: every change in DV01; a roll to a new instrument as two outright one-ways; a quarterly re-strike of each par leg. *Why:* the conservative end, since a calendar spread trades tighter.
- **K4**. The breakeven cost is closed-form, c* = 2 × gross P&L / DV01 traded, or "none" where gross ≤ 0.
- **K5** (2026-09-28). The hysteresis pair (enter 1.0, exit 0.0) was fixed in config before the grid ran, and is not chosen from it. *Why:* the grid is there to be shown, not tuned on. The best cell is not adopted.
- **K6**. The ELB state is the policy rate on its floor with the rule below it, where the model has no view. Three treatments: flat (the book's), exclude, and hold.
- **K7**. Sleeves are vol-scaled for evaluation (each alone at the book's risk); unit sizing is secondary. *Why:* unit-DV01 statistics weight the high-vol years, and the sizing decides the sign of the sum.
- **K8**. A 10% no-trade band on the vol multiplier. *Why:* rescaling fell from 22% to 12% of DV01 traded.
- **K9**. Sharpe is daily mean over sd × √(observed sessions a year); its SE is √((1 + SR²/2) / years).
- **K10**. The carry filter stays a diagnostic (`positions.carry_filter: false`). *Why:* turning it on after seeing the result would be tuning.
- **K11**. The brief's trades table shows each sleeve's round trip.
- **K12**. The stale "no transaction costs yet" tag is replaced by the asymmetry: USD costs are tick-based, GBP costs assumed.
- **K13**. **The post-cost result.** Vol-scaled, the equal-risk sum is -0.09 gross and -0.53 net: there is nothing for costs to halve. One sleeve of five clears its costs, the GBP outright (+0.62 net, breakeven 3.4bp against an assumed 1bp), and 2022-23 make all of it. The USD outright loses before costs. 56% of what the sleeves trade is maintenance. Nothing was done in response: no cost loosened, no signal changed, no grid cell adopted.

## Model variants and robustness (2026-09-28)

V1-V11's numbers are from runs to 2026-09-21; the current comparison is `reports/robustness.md`.

- **V1**. A model-side variant is a config override that reaches every stage, and the merged block must pass the schema. *Why:* before this an r* or curve override could be silently dropped, making a robustness row the baseline twice.
- **V2**. HLW r* (USD only), from the NY Fed's real-time vintages, with Taylor's 2% before the first. corr(z) with the baseline 0.96.
- **V3**. Constant r*: USD 1.1 (the SEP r*'s mean), GBP -2.6 and -0.6 (±1pp). The z absorbs most of a constant r*, not all: the floor binds on some sessions and not others.
- **V4**. Estimated coefficients: a partial-adjustment fit, real time, shrunk toward the imposed (0.5, 2.0) with the prior worth two years. *Why:* this is the evidence for R1: the estimate wanders far from the imposed values and moves with episodes more than it settles on a reaction function. `model/estimate.py`
- **V5**. z windows of 365, 548 and 1095 days around the chosen 730, with `min_periods` scaled.
- **V6**. GBP Nelson-Siegel-Svensson curve fitting on a grid; marks stay on the Bank's curve. corr(z) 0.999.
- **V7**. Converge-to-target conditioning: inflation and the gap halve their distance to target every four quarters.
- **V8**. The OBR u* series is deferred; GBP u* stays 4.5%. *Why:* it means transcribing about 30 forecasts by hand, and the r* ±1pp rows already bound a u* ±0.5pp error.
- **V9**. The estimated rule also drops a quarter ending on the floor after a cut that quarter. *Why:* the March 2020 emergency cuts imply a goal of -8.4%, which a floored rule can never have, and that one row moved the estimate more than any other.
- **V10**. Under converge conditioning, the ELB state is where the converge path is flat on the floor. *Why:* otherwise the flat treatment would flatten a view.
- **V11** (2026-09-29). The robustness grid: 13 rows over 8 choices, one at a time, declared in `config/strategy.yml`. A row changes only the signal; marks, instruments, carry, costs and covariance are the baseline's, checked on every row. The chosen value of every choice was fixed before the grid ran.
- **V12** (2026-09-29). The cells: the headline book's net Sharpe over a common sample, each sleeve alone, and IC(21). Every move carries a paired SE. *Why:* rows share sessions and most positions, so a cell's own SE is the wrong yardstick for a difference.
- **V13** (2026-09-29). **The robustness reading.** Every row loses money net, and none moves the book by as much as one SE. Four rows move it by one to two paired SEs, all upward, about what chance gives. The GBP outright earns in every row and the USD outright loses in every row. The GBP outright depends on the imposed coefficients. Nothing is re-chosen in response.
- **V14** (2026-09-29). **The note on r\*.** A constant GBP r* moves the quoted gap 27.7bp per pp away from the floor, and the z absorbs it only where no goal has been floored in the trailing window, a small part of the traded sample. For the book the constant is nearly free; for the GBP outright it is not quite. The r* tag, once "moves the bp gap, not the z", overstated this; it now reads "moves the bp gap and mostly leaves the z unchanged, except near the floor".

## The Fed's projected path as the reference (2026-09-30)

- **V15**. A variant, `rule.projected`, swaps the rule's path for the Fed's own SEP median path (FEDTARMD, ALFRED vintages). It is not a grid row: it would move the common sample to 2016. *Why:* a diagnostic found the rule closes the gap toward the market, mostly through the policy rate, so the Fed's own projection was the natural reference to test. No frozen number moves.
- **V16**. The projected path is linear in time from the rate in force through each year-end projection. *Why:* the SEP says nothing about meetings in between; linear is simplest.
- **V17**. The GBP - USD differential keeps every session both currencies have, NaN where a gap is missing. Nothing moves.
- **V18**. **The reading: the dots do not rescue the USD outright, and confirm that the USD market leads the Fed.** Against the dots the USD outright nets worse (-0.82 against -0.61) and its IC(21) falls to -0.27. The dots move toward the market (+0.68 per bp, t +4.7), not the reverse (+0.09, t +0.5). Trading the gap the other way would be a sign chosen after the result; it needs registering first.

## Portfolio (2026-09-29)

Numbers are the headline book (inverse-vol, shrunk, every sleeve, no drawdown control) at configured costs.

- **P1**. The book calendar is USD's sessions; a sleeve's P&L is credited to the next book session. *Why:* one calendar gives one covariance and one set of decisions.
- **P2**. The covariance is a zero-mean EWMA (λ 0.97) of the sleeves' unit P&L, strictly before the close, on complete rows.
- **P3**. The one-lag term carries the Bartlett weight (Newey-West at one lag). *Why:* the unweighted sum, S0 + S1 + S1', was not positive definite on 905 sessions. With the weight it is positive definite on all.
- **P4**. Shrinkage toward the diagonal with a Ledoit-Wolf-type intensity (averages 0.20). It halves the median condition number and matters only for mean-variance.
- **P5**. Each sleeve is sized on the sd from the covariance's diagonal, floored as in K7.
- **P6**. Constructions size each sleeve's hysteresis side: inverse-vol, ERC, and mean-variance with μ = side × min(|z|, 3) × sd.
- **P7**. Gross DV01 is capped at 0.4% of capital per bp. *Why:* a stress limit from arithmetic: one 25bp step against every leg loses 10%. Uncapped, vol targeting sized up to $1.5m per bp in 2014.
- **P8**. The vol target (5% ex ante) is the whole book's, whatever number of sleeves has a side.
- **P9**. The 10% no-trade band applies to each leg; the gross cap outranks it.
- **P10**. Turnover is split by cause: entries and exits, resizing, rescaling, rolls, re-strikes.
- **P11**. The drawdown control (halve at 10%, restore under 5%) is reported beside the book, not used in it. *Why:* it fired once and never released; one episode cannot support a rule.
- **P12**. A relative-value-only book (no outrights) stands in for duration neutralisation. It loses before costs.
- **P13**. The book decides from 250 complete sessions (2012-03-01) and is counted from Jan 2014, when the flat ELB treatment first allows a position.
- **P14**. The headline book is inverse-vol, shrunk, every sleeve, no drawdown control. *Why:* explainable in two sentences, its weights do not depend on a noisy covariance, and it was set before any book ran. ERC cannot be told from it; mean-variance is worse.
- **P15**. **The book's result.** The headline loses before costs and after: gross -0.15, net -0.62 (0.32), worst drawdown 57.5%. What earns is one window, Dec 2021 to Aug 2023 (+20.5% of capital inside it, -59.2% outside). The GBP outright and 2022 are one source of return, not two. Nothing was tuned in response.
- **P16**. Two books are compared on the paired SE of their Sharpe difference (Jobson-Korkie with Memmel's correction). *Why:* books that share most positions make a single book's SE overstate the noise of a difference 3 to 15 times.
- **P17**. **A diagnostic.** The covariance is scored as a forecast of the unit P&L it sizes (after Paleologo, ch. 5). Realized runs above ex-ante vol mostly from vol dynamics the EWMA does not track; no candidate beats the configured covariance by two DM t. Nothing changed.

## Attribution and evaluation (2026-09-29)

Numbers are the headline book net of costs, from `reports/metrics.md`.

- **A1**. Performance beyond Sharpe: daily and per-trade hit rates, trades, worst drawdown, turns and time in market. A trade carries the costs of its entry and exit.
- **A2**. `metrics.ex_2022` and `metrics.ex_2022_23` leave the window's P&L out without re-running the positions. *Why:* one well-telegraphed cycle can carry a rates backtest, and "what did the rest earn" is the question.
- **A3**. P&L by component group (level, slope, cross-country), split into carry, roll and rate exactly as the expression layer marks it (C9).
- **A4**. **Level or relative value.** Each sleeve's P&L is split into its exposure to the outrights and the rest. The level is 63% of the variance of the book's daily gross P&L, so the book is closer to a duration timer than to relative value; but the loss is mostly the relative value's.
- **A5**. Carry against rate: the book pays more in carry and roll than its rate calls earn. A carry benchmark (the same book trading the sign of carry) correlates -0.16 with it: not a carry trade.
- **A6**. Regime conditioning, real time. Every regime Sharpe is within about one SE of the others: descriptive, not a test.
- **A7**. **The ELB treatment is flat, with the alternatives shown.** Exclude is not a strategy; hold loses 17% of capital through six years of GBP floor. All three are within a paired SE or so.
- **A8**. The first z's after lift-off are standardised against a window mostly of ELB sessions. Reported as a limitation, not corrected.
- **A9**. IC at 5, 21 and 63 sessions outside the ELB, with a non-overlapping check. Only the GBP outright has a signal; the USD outright has the wrong sign. The t overstates precision because z is persistent.
- **A10**. The tear sheet is one generated page; it raises rather than spill past a page.
- **A11**. On the Windows machine the series added for the strategy layer (L12-L17) were pulled with `update_data.py --only`, and the currency-branch test was fixed to compare POSIX paths. Regress references differed in the last bits; nothing was refrozen (see A18).
- **A12**. Where the costs go: cost = turns × mean gross DV01 × one-way cost, checkable by hand (19.8 × 284k × 0.44bp = 2.47% of capital a year; 46% maintenance). *Why:* an outside review suspected a double count. The USD outright's 2018, rebuilt contract by contract (58,462 contracts), costs $1,276,426, the cost charged to the dollar. Rolls at half price would save about 0.4% a year and gross would still be below zero.
- **A13**. **Carry against rate, by sleeve.** Right and bleeding: GBP outright, USD 2s10s. Wrong and bleeding: USD outright, GBP 2s10s. Wrong and collecting carry: GBP - USD 2y. The USD outright's loss is its rate calls, not its carry.
- **A14**. What the rate calls would have had to earn: the book's rate change was about a third of what a net of zero needs. Only the GBP outright's correlation clears the needed one.
- **A15**. The carry filter skips 16 of 104 entries; the book nets -0.52 against -0.62 (+0.10, paired SE 0.05), still losing. It stays off. *Why:* most carry is paid while positions are held, not at entry.
- **A16**. **The level, as a measurement.** USD is below the rule in all 13 years (-25 to -86bp at the fourth meeting); GBP in 8 of 11, above from 2024. *Why:* a discount to the dots is a known pattern, so this is a replication with a rule; what the level is (term premium, the rule's error) is not identified.
- **A17**. Wording: the tear sheet adds IC and the drawdown's shape; the brief's cross-country row shows only z, since its bp level is mostly the two r* choices.
- **A18** (2026-09-30). One fixture reference for both machines: checked to `regress.PLATFORM_ULPS` (4096 ulps of each column's largest value) instead of bit for bit; the full tier stays bit for bit. *Why:* the Mac and Windows machines should agree on one reference. Compilers and BLAS libraries round differently, and a per-platform reference would let the two drift.

## Credit bridge (2026-09-28)

D1-D3 were written before `credit.py` was run on the data; everything after them was written from the run.

- **D1**. **Pre-registered before test 2 ran.** Expected sign: negative. A gap below zero (tightening underpriced) predicts a widening of Baa - Aaa. Headline cell: Baa - Aaa on the fourth-meeting z, h = 63, every session, Newey-West lag 63. Supportive only if the slope is negative with NW t ≤ -1.96 and the non-overlapping mean slope is negative too. Every other cell is a sensitivity, not a second chance. *Why:* writing the sign first stops it being chosen after the fact. (After the run, D18: the second condition adds little.)
- **D2**. **Fixed before the check ran.** Staleness shows if a spread's weekly AR(1), or its correlation with last week's 10y change, is outside ±1.96/√n; then the 10y change is a control.
- **D3**. **Stated before test 3 ran.** The slope of weekly Δ(Baa - Aaa) on the level's change is negative in early hiking and positive in late hiking.
- **D4**. Spreads: primary Moody's Baa - Aaa (maturity-matched); Baa - 30y and Baa - 10y alongside; ICE IG and HY OAS from 2023 as a cross-check. Spread changes stand in for excess returns. The ICE series are only on FRED for three years.
- **D5**. Tests 1 and 3 use weekly changes to each Wednesday; test 2 uses the signal's sessions.
- **D6**. The level's weekly change holds the meeting that was fourth at the start of the week. *Why:* re-choosing would book a passing meeting as a repricing.
- **D7**. Newey-West Bartlett standard errors; test 2's cross-check fits every h-th session over all start offsets.
- **D8**. The regime state machine (`regimes.py`), real time: ELB, early hiking (first 12 months), late hiking, holds, cutting. Round-number thresholds, not fitted.
- **D9**. ELB sessions are in test 2's headline, as registered, and out as a row.
- **D10**. The 2020 crash is a sensitivity window. Without it the headline fails D1's rule.
- **D11**. Baa - Aaa does not trip the staleness check; the Treasury-benchmarked spreads do, so they carry the control.
- **D12**. **Test 1: a null.** The policy path's repricing explains 0.3% of weekly Δ(Baa - Aaa).
- **D13**. **Test 2: the registered sign, narrowly, and not robust.** -2.20bp per unit of z over the next quarter, NW t -2.09: it passes D1's rule. It fails without the crash, without ELB sessions, at h = 21, without 7 of 15 years, and under either other kernel; a placebo puts its one-sided p at 0.07. Weak evidence at the edge of chance; not a lead to trade.
- **D14**. **Test 3: no demonstrated sign flip.** The sign differs in one of the two cycles and not the other.
- **D15**. **Power is too low for a sign-flip test.** The minimum detectable difference is about five times the estimate; the sentence says "differs / does not differ in each of two cycles" and nothing stronger.
- **D16**. Method constants live in `credit.py`; currency data lives in its `credit` config block.
- **D17**. **Added after the run, not registered:** test 2 without each calendar year. The slope stays negative in all 15 but fails the rule without 7; it leans most on 2014 (overlapping the oil collapse) and 2020.
- **D18**. **Added after the run, not registered:** four more checks. Bartlett at lag 126 (t -1.81) and Hansen-Hodrick (t -1.72) both miss the line; clipping z or dropping floored-sd sessions makes it stronger. D1's cell and verdict stand.
- **D19**. **Added after the run, not registered:** a time-rotation placebo. The registered t reaches ±1.96 in 23% of shifts against a nominal 5%. The reading changes to "at the edge of chance"; the verdict stands as registered.
