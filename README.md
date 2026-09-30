# GlobalPolicyEngine

How well do markets price the path of central banks? This project is built to isolate these differences around currencies and trade them. USD and GBP are built. A third currency waits on a data purchase; the reasoning is in [notes/DECISIONS.md](notes/DECISIONS.md) (S1).

The short answer so far: the gap between the market's path and the Fed's and the Bank's own rule is measurable, and it closes in the direction you would expect. Once it is traded as a five-sleeve book with carry, roll and costs counted, it loses money: a net Sharpe of -0.62 (SE 0.32) over 2014-26, with 2022-23 the only stretch that pays. The one sleeve that earns in every specification is the GBP outright.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/implied_paths_USD_dark.png">
  <img alt="Implied fed funds path from ZQ futures on the first session of each month since 2010, against realized EFFR, and the implied move by meeting horizon as a heatmap" src="reports/figures/implied_paths_USD_light.png">
</picture>

## Status

Working end to end for USD:

- A cache under `data/` that one command rebuilds from nothing: ZQ, SR1 and SR3 settlements out of a Databento GLBX.MDP3 archive on disk, and EFFR, SOFR and the target range from FRED. Every observation carries a reference date and a publication date, and a revised value is a new vintage, not an overwrite. A revision the source can't date, and no longer serves once replaced, exists only in this cache, so `scripts/backup_cache.py` snapshots it after each update.
- The FOMC meeting calendar, scraped and committed, with announcements from 2010-01-27 through 2027-12-08 (146 meetings). Each has an announcement date, an effective date, a `scheduled` flag and, for the 17-18 March 2020 meeting the Fed brought forward to the Sunday, the day it was called off. Each session sees the calendar as it stood that day: the two March 2020 emergency cuts are pillars only from their announcement, and the meeting they replaced is one until it was cancelled.
- A piecewise-constant EFFR path over the next eight meetings on every ZQ session from 2010-06-07 to 2026-09-21: 4,110 sessions, no solver failures, 99.68% of Fed business days. Of the 13 missing days, eight are Good Fridays, three are Fridays CME closed for a Saturday holiday, and two (2020-02-27, 2020-06-30) are days Databento flags as degraded, with no ZQ settle in the archive. The coverage report lists them.
- A SOFR discount curve bootstrapped in QuantLib from SR3 futures and SOFR fixings, and a cross-check of the ZQ path against SR1 (below).
- A real-time macro nowcast on every Fed business day from 2011-03-04, the first day every input has an ALFRED vintage (CBO's natural rate from 2011-02-02, average hourly earnings from 2011-03-04).
- A model path on each of the 3,921 sessions since then: the Fed's own balanced-approach rule from the Monetary Policy Report, in its inertial form, fed by that day's nowcast, with r* from the FOMC's longer-run dot and an explicit floor at the lower bound. The gap between the two paths, per meeting and z-scored on its trailing two years, and a crude backtest of it (below).
- GBP through the same interface (below): the Bank of England's fitted OIS curve on every London business day from 2009-08-03 (4,332 sessions, no failures), a real-time UK nowcast from the ONS's own revisions triangles from 2010-08-26, and the same rule. The GBP path lands on the Bank's own MPR conditioning paths to 1.0bp on average over 29 reports.
- A weekly one-page brief, generated from cache: both currencies' paths, the gaps, the GBP - USD differential, what changed since last week and why.
- Every currency-specific value in `config/currencies.yml`, validated against a schema when it loads, and a test that fails if a shared module names a currency in code. Adding a currency is a config block, a meetings file, and a source module only if its data comes from a new provider.
- A regression harness (`scripts/regress.py`) that freezes every stage's output and checks it bit for bit. Both currencies' full samples are identical to week 5 after the week 6 refactor, and again after weeks 7-12. The look-ahead tests run per currency.
- Five sleeves that trade the gap (below): the USD and GBP outrights at the fourth meeting, USD and GBP 2s10s on an orthogonalised slope, and the GBP - USD 2y differential. Each has real instruments (ZQ contracts, OIS forwards, par Treasuries and gilts), carry and roll with three independent checks on the P&L, and a breakeven that says, at the close, whether a signal pays for its bleed.
- Costs, turnover and a hysteresis rule for every sleeve, with a stated treatment of the lower bound. Then one book, all five sleeves sized together under a shrunk covariance and a 5% vol target, and a robustness grid that moves one choice at a time off the chosen specification.
- A credit bridge for USD: three tests of the gap against Baa - Aaa and four other spreads, with the predictive test's sign registered before the first run.
- A weekly one-page brief now carries each sleeve's trade and its carry and roll.
- Every choice so far, with its date and reason, in [notes/DECISIONS.md](notes/DECISIONS.md). The ones left to the author are marked *proposed* and listed in [notes/author_review.md](notes/author_review.md).
- `uv run pytest -q` gives 944 passed, 6 skipped.

Not built yet: the tear sheet and metrics (week 10), the written note (week 11), and the generated limitations section and final hygiene pass (week 13). The plan and where it stands are in [notes/build_spec_w7_w13.md](notes/build_spec_w7_w13.md).

## Install

Python >=3.11 (developed on 3.13). Dependencies are `pandas`, `pyarrow`, `databento`, `requests`, `pyyaml`, `matplotlib`, `openpyxl` and `QuantLib`, the last pinned to an exact version because its bindings change signatures between releases. `beautifulsoup4` is dev-only, used by the FOMC scraper, which never runs inside the test suite.

```bash
uv sync
```

The package is src-layout (`src/policypath/`) and is installed editable by `uv sync`, so it is imported as `policypath`, never off `sys.path`.

## Quick start

The test suite runs on committed fixtures and needs neither the network nor the Databento archive:

```bash
uv run pytest -q
```

Rebuilding everything takes two commands. The first needs the archive (paid, gitignored -- see [Data](#data)) and a free FRED key in `.env`. A first build takes about thirteen minutes; after that it is incremental, and a second run adds nothing:

```bash
uv run --env-file .env python scripts/update_data.py
```

The second reads only the cache. It writes the path panel to `data/panel/`, and the coverage report, the SR1 cross-check and the figures to `reports/`:

```bash
uv run python scripts/build_panel.py
```

One session, solved exactly as the panel solves it:

```bash
uv run python scripts/run_implied_path.py 2024-09-17
```

The macro side needs only the FRED key, not the archive. Pull every ALFRED vintage, then catalogue what each series covers and build the real-time nowcast from the cache (`reports/vintages_USD.md`, `reports/nowcast_USD.md` and its figures):

```bash
uv run --env-file .env python scripts/update_data.py --macro-only
uv run python scripts/catalogue_vintages.py
uv run python scripts/build_nowcast.py
```

One date's nowcast, as it could have been read that evening:

```bash
uv run python scripts/run_nowcast.py 2024-01-20
```

With the panel and the nowcast built, one more command writes the model path, the gap signal, the backtest, `reports/model_USD.md`, its figures and the dated one-page note `reports/onepager_USD_<session>.pdf`. Another prints one session's market and model paths side by side:

```bash
uv run python scripts/build_model.py
uv run python scripts/run_model_path.py 2021-11-01
```

GBP needs no archive, but its update needs the FRED key too, for the dollar-sterling rate (DEXUSUK). Pull the Bank of England's curves, SONIA and Bank Rate and the ONS vintages, then build the panel, the nowcast and the model, and check the path against the Bank's MPR conditioning paths:

```bash
uv run --env-file .env python scripts/update_data.py --ccy GBP
uv run python scripts/build_panel.py --ccy GBP
uv run python scripts/build_nowcast.py --ccy GBP
uv run python scripts/build_model.py --ccy GBP
uv run python scripts/check_mpr.py
```

With both currencies' panels, nowcasts and models built, the strategy layer reads only `data/panel/` and the cache. The expression and carry come first, since the brief's trades table reads them. Then the credit bridge:

```bash
uv run python scripts/build_expression.py
uv run python scripts/build_credit.py
```

The weekly brief, for every currency in `config/brief.yml`, one page to `reports/brief_<date>.pdf`. `--date` gives the brief that could have been sent on a past day:

```bash
uv run python scripts/build_brief.py
```

Costs and the hysteresis grid, per sleeve (`reports/costs.md`):

```bash
uv run python scripts/build_strategy.py
```

The book, then the robustness grid. The grid stops unless its baseline equals the book's headline in `reports/results/portfolio.json`, so the book goes first. A first grid build rebuilds every row's signal (about 100s); after that it reuses the ones whose inputs have not changed, and `--rows` rebuilds only the rows it names:

```bash
uv run python scripts/build_portfolio.py
uv run python scripts/build_robustness.py
```

Before changing anything that computes, freeze every stage's output over the full sample, then check after each change. A check stops at the last session the freeze covered, so an updated cache compares like for like. It exits non-zero on any difference, however small. The committed fixtures have their own reference, which the test suite checks:

```bash
uv run python scripts/regress.py freeze
```

```bash
uv run python scripts/regress.py check
```

## How the path is solved

`policypath.curves.policy_path.implied_path` takes the implied average rate per contract month (`100 - price`) and the meeting effective dates, and returns the overnight rate in force in each regime between them.

A ZQ contract settles to the arithmetic average of EFFR over every *calendar* day of its delivery month. So each contract month is one linear equation in the regime rates, with coefficients equal to the share of the month's days that each regime covers. The solver builds that weight matrix (months x regimes) and solves all months jointly by least squares.

Mid-month, part of the front contract is already history. Days whose EFFR is published enter the front month's average as known numbers, and the path starts on the first day that is not. On session t that is t itself: the NY Fed publishes t - 1's fixing on the morning of t, and the market settling on t did not know t's own. `calendars.known_daily` turns published fixings into known calendar days, so weekends and Fed holidays carry the last fixing and a Columbus Day session knows only through the Thursday.

Substituting history exposes one weakness. When the next meeting is days away, the rate from today until then lives in a few days of the front contract, and price noise in that contract is multiplied by days-in-month over days-left. On 2024-09-17, two days before the cut took effect, that put today's rate 10bp away from the 5.33% EFFR actually fixing. So a first regime shorter than 10 days is pinned to the last fixing. Across the 46 announcement days since 2021, where the decision is known before ZQ settles, that took the error in the implied step from 5.8bp mean and 34bp worst to 0.5bp and 1.6bp. The more obvious fix, dropping the front contract when it is near expiry, gained nothing once short regimes were pinned, and on its own it only moved the amplification into the next contract. It is off.

## Checking the path against SR1

One-month SOFR futures are an independent market on a different overnight rate, and SR1 settles on SOFR exactly as ZQ settles on EFFR: the average over every calendar day of the contract month. For every session, `curves/basis.py` averages the ZQ-implied EFFR path over each SR1 month, using realized SOFR on the days already fixed. It then solves for the constant spread that reprices the SR1 settle: the SOFR - EFFR basis the two markets imply together. A wrong ZQ path would make that spread jump around at random.

Instead, over 2,108 sessions from SR1's first trade (2018-05-07), the basis for the first month wholly ahead moves 0.57bp a day (sd), and sits 1.88bp on average from the basis that later printed. In 2020-24 that gap is 0.85-1.42bp a year. It is wider where funding is known to have moved: the SOFR month-end spikes of 2018-19 and the September 2019 repo spike, and the 2025 squeeze, which the market priced late and then expected to last. Against the SR3 quarters this check used to run on, over 2021-24 when both are available, SR1 is closer to the realized basis in three of four years (0.85-1.42bp against 1.09-1.46bp; 2022 is the exception) and smoother in every one (daily sd 0.30-0.48bp against 0.43-0.71bp). The same code runs on SR3 by setting `sofr.crosscheck` in `config/currencies.yml`; SR3 stays what the SOFR discount curve is built from, since SR1 lists only about 13 months out.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/sofr_basis_USD_dark.png">
  <img alt="SOFR minus EFFR basis implied by SR1 against the ZQ path, and the basis realized over the same days" src="reports/figures/sofr_basis_USD_light.png">
</picture>

## Against the Fed's own rule

`model/` puts a second path next to the market's on every session: where the balanced-approach rule the Fed publishes in its Monetary Policy Report would take policy, given only what was public that day. The rule is imposed rather than estimated, R* = r* + π + 0.5(π − 2) + 2(u* − u), and approached at the MPR's inertial pace (0.85 a quarter, so 0.922 a meeting). π is the nowcast's 12-month core PCE and u* is CBO's natural rate, both as vintaged on the day. r* is the median longer-run funds rate from the latest Summary of Economic Projections, minus 2%. Each SEP is a dated document that is never revised, so this is real-time with no vintage machinery. The rule's notional rate is floored at the lower bound in a named step, and the unfloored value is kept, so the 977 sessions where the rule asks for a negative rate stay visible. A path eight meetings out needs the macro state eight meetings out; rather than forecast it, the model holds today's inflation, gap and r* flat. It answers the question "if nothing changes, where does the rule take policy?".

The gap is market minus model in bp at each meeting. It widens where it should: late 2021 (−159bp at the eighth meeting as the rule left the floor and the Fed waited), the 2019 insurance cuts, the 2023 pivot debate and the September 2024 cut repricing, and it sits near zero while both are on the floor. `tests/test_no_lookahead.py` runs the whole chain (market path, nowcast, r*, rule, gap, z, P&L) on inputs truncated at D. It then appends everything after D and poisons it, and checks nothing at or before D moves. A 200-day SEP leak and a one-day fixings leak each fail it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/model_vs_market_USD_dark.png">
  <img alt="ZQ-implied path against the balanced-approach rule's path over the next eight meetings on the latest session, and the gap at the fourth meeting ahead since 2011" src="reports/figures/model_vs_market_USD_light.png">
</picture>

The crude backtest trades the implied rate at the fourth meeting ahead on that meeting's z. It receives when the market is above its usual relation to the rule, holds DV01 fixed and trades one session after the signal. Over 2012-26 it returns a Sharpe of 0.02 **before costs**: it loses through 2014-20, when the market priced lower-for-longer and then cuts the rule never justified, and makes it back in 2021-25. The horizon was fixed before any P&L was run; the others are in `reports/model_USD.md` as a sensitivity, not a menu.

## GBP: the Bank of England's curve

There is no ZQ outside the US. What makes the US extraction closed-form (a contract settling on the arithmetic monthly average of the overnight rate) has no European counterpart, so GBP comes from a different mechanic behind the same interface. `market.path(date, ccy)` returns the same object for both currencies, from the backend the config names. The US panel is bit-identical to week 4 through it.

The Bank publishes a fitted OIS spot curve every day, continuously compounded, at monthly maturities from one month to five years. The expected SONIA between two meetings is the forward over that window, exact from two spot rates. Meeting dates fall between maturities, so the log discount factor is interpolated linearly between them. Today until the first meeting is pinned to the rate in force: the last SONIA fixing, moved by any Bank Rate change since, because a decision applies from noon on the day it is announced. Against the SONIA that then printed, pinning cuts the error on that first regime from 0.84bp to 0.52bp (2022: 3.3bp to 0.1bp).

The check is the Bank's own. Each MPR projection is conditioned on a Bank Rate path averaged from the same OIS curve over 15 working days. Rebuilt from the step path for all 29 reports from August 2019 to July 2026, it lands within 1.0bp on average (August 2024: 0.36bp at worst). The differences have a sign you can predict. The Bank averages a smooth spline, which starts a priced move before its meeting; the step path moves on the day. So the difference correlates −0.59 with the move priced inside the quarter.

The UK labour data is the weak input, and the brief says so rather than smoothing it. LFS unemployment, which the gap is built from, was suspended in late 2023 and rebadged; the last official vintage is held through the suspension. Read as first printed against the claimant count and PAYE payrolls, it has agreed less since 2023: the LFS and claimant 12-month changes correlated +0.72 to 2022 and −0.16 since (`reports/nowcast_GBP.md`). There is no published UK longer-run policy rate, so r* is a constant. So is u*. The traded z subtracts the gap's trailing mean, so a constant moves the bp level, not the trade.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/model_vs_market_GBP_dark.png">
  <img alt="Bank of England OIS-implied Bank Rate path against the balanced-approach rule's path, and the gap at the fourth meeting ahead since 2010" src="reports/figures/model_vs_market_GBP_light.png">
</picture>

The GBP crude backtest (same expression, fourth meeting, no costs) returns a Sharpe of 0.75, but 90% of it is 2022-23. One regime again, and no evidence yet.

## From a gap to a trade: carry and roll

`strategy/` turns each signal into positions in instruments you could hold. A unit is +1 USD of DV01 on the sleeve's first leg; positive means receive. z > 0 always means the market prices more tightening than its reference, so the sleeve receives. The USD outright holds the ZQ month that the fourth meeting's regime settles in, with its DV01 derived from the contract spec, never typed in code. The GBP outright holds the OIS forward over that regime. The curve and cross sleeves hold par Treasuries and gilts, struck at each close. Every leg's P&L splits into carry, roll and the rate's move. Three checks guard it: the split must add up on every session; an independent full revaluation from each instrument's own price must land inside a bound set by convexity (0 sessions outside it); and synthetic convergence tests pin the direction. The build refuses to write the report if any of the three fails.

The finding (`reports/expression.md`): **a correct signal can still be a losing trade.** Of the 3,938 signal sessions whose next quarter went the signal's way on the rate, 19% still lost money once carry and roll were counted. On 53% of signal sessions, carry and roll ran against the side before the rate moved at all. Whether that bleed is a reason to stand aside turns on 2022. Over the full sample, the bleeding signals earned more, not less. Leave out every quarter that overlaps 2022 and they earned less. So the carry filter stays a diagnostic, off in the book.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/carry_dark.png">
  <img alt="Signal sessions that were right on the rate but lost once carry and roll were counted, by sleeve" src="reports/figures/carry_light.png">
</picture>

## Net of costs

Every sleeve runs under a hysteresis rule: it enters at |z| >= 1 and exits when z crosses zero. The pair was fixed in config before any grid ran. The book stays flat while the policy rate and the rule are both at the lower bound, because the model has no view there. Costs are charged on every change in DV01, on every roll into a new futures month or forward window, and on each quarterly re-strike of a par leg. ZQ is costed at two back-month ticks plus the fee, 1.048bp a round trip. That is a tick, not a measured spread: the archive holds settles, not quotes. Every other leg's cost is assumed (0.5bp for Treasuries, 1bp for gilts and the OIS forward), since those legs are valued off fitted curves.

The verdict in `reports/costs.md` is *fragile*. Each sleeve at the book's risk, the sum nets -0.52 (0.31) and is already negative before costs (-0.09). Only one of the five sleeves clears its costs: the GBP outright, +0.62 (0.36) net, with a breakeven of 3.4bp against its assumed 1bp. 66% of its net P&L is 2022, and 2022-23 make all of it. Most of what the sleeves trade is maintenance, not signal: 56% of DV01 traded goes to rolling onto the new fourth meeting and re-striking par legs.

## The book

`strategy/portfolio.py` sizes the five sleeves together. The covariance is an EWMA of the sleeves' unit P&L (lambda 0.97). A Newey-West lag term handles London closing before New York. The matrix is shrunk toward its diagonal, which cuts the median condition number from 11.3 to 5.7. The headline construction is inverse-vol, marked *proposed* for the author. It targets 5% ex-ante vol on USD 100m, with gross DV01 capped at 0.4% of capital per bp. Risk parity (ERC) and mean-variance run beside it. Books that share most of their positions are compared by the paired SE of the difference, not by either Sharpe's own SE.

The result (`reports/portfolio.md`): **the book loses money, net of costs and before them.** The headline book's gross Sharpe is -0.15 (0.29), and it nets -0.62 (0.32), -3.27% of capital a year, with a worst drawdown of 57.5%. No cost makes it work, because the gross is already below zero. ERC and mean-variance differ from it by less than two paired SEs. The book's P&L comes almost entirely from Dec 2021 to Aug 2023: +20.5% of capital inside that window, -59.0% outside it. Take out the outrights and the relative-value book still loses before costs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/book_equity_dark.png">
  <img alt="The headline book's equity, gross and net of costs, with each sleeve's contribution" src="reports/figures/book_equity_light.png">
</picture>

## Robustness

`strategy/robustness.py` moves one choice at a time off the chosen specification and rebuilds the signal through the chain's own entry points. The 13 rows cover r* (HLW in real time, constants), estimated rule coefficients, the z window, NSS curve fitting for GBP, converge-to-target conditioning, the raw slope, and the carry filter. The marks, instruments, costs and covariance stay the baseline's, and a check on every row confirms it. Every row counts the same 2,866 sessions.

**No single choice turns the loss into a gain** (`reports/robustness.md`). The 13 rows net from -0.71 to -0.32, around the chosen -0.62. None moves the book by as much as one SE of its Sharpe, and none by more than two paired SEs. The USD outright loses in every row and the GBP outright earns in every row. The GBP outright's rank IC with its next 21 sessions is +0.40 at the chosen specification (non-overlapping range +0.33 to +0.46). The chosen specification was fixed in config before the grid ran, so it is the reference, not a cell picked from it.

## The credit bridge

`credit.py` asks whether the policy-path gap says anything about USD credit. The primary spread is Moody's Baa - Aaa, which is maturity-matched. Baa against the 30y and 10y Treasury and ICE BofA's IG and HY OAS run alongside. Spread changes stand in for excess returns. The expected sign of the predictive test, and the rule it has to pass, were registered before the first run (D1).

The results (`reports/credit_USD.md`). Contemporaneously, a repricing of the policy path explains 0.3% of the weekly variation in Baa - Aaa. Predictively, a unit of z at the fourth meeting comes before Baa - Aaa narrowing 2.2bp over the next quarter, NW t -2.09. That passes the registered rule, but only just. It fails under Hansen-Hodrick weights (-1.72) or Bartlett at lag 126 (-1.81). It fails without 7 of the 15 years. A time-rotation placebo gives a one-sided p of 0.07. So it sits at the edge of chance, and the report says so. The conditional test (early against late hiking) has the hypothesised sign in one cycle and not the other, well under its minimum detectable difference.

## Data

- **`databento/`** is gitignored and not distributed: it is a paid batch archive, ~1.2 GB, one definition and one statistics file per day per Databento batch job. Six jobs cover ZQ from 2010-06-06 to 2026-09-21, SR3 from 2020-12-31 to 2026-09-21, and SR1 from 2010-06-06 to 2026-09-24 (it first listed in 2018). Every job names its daily files the same way, so each is kept in its own folder, `databento/<definitions|statistics>/<job_id>/`, with its manifest, metadata and condition files. Download a new job straight into `databento/definitions/` or `databento/statistics/`, then run `uv run python scripts/file_databento.py`. It files every file under its job by the sha256 in the job's manifest, so the ` (2)` copies a browser makes of colliding names do not matter, and it checks every job is complete. `sources/archive.py` knows which job holds which root, and only `sources/` reads the archive.
- **`data/`** is gitignored and entirely derived. `data/cache/` holds one parquet vintage log per `(source, currency, series)` plus `manifest.json`, which records the ranges already covered and is what makes `update_data.py` incremental. CME sends a preliminary settle around 16:00 ET and the final that evening, or on the Sunday for a Friday session. Both are kept, so a read at the close sees the preliminary. The panel uses each session's settles as they stood by the end of the next business day, and records when the last one arrived.
- **FRED** (free key) supplies the rest of the USD side: EFFR, SOFR and the target range, ALFRED vintages for the macro inputs, H.15 Treasury par yields for the Treasury legs, Moody's Baa and Aaa for credit, and DEXUSUK for sterling. ICE BofA's OAS series are on FRED only for the last three years (from 2023-09-29 as the cache stands), so they are a short cross-check, never the headline.
- **Bank of England and ONS** data need no key: `sources/boe.py` (IADB series, the OIS curve archive, the nominal gilt curve for the gilt legs) and `sources/ons.py` (revisions triangles, the release calendar). The MPC calendar is scraped by `sources/pull_mpc.py` into the committed `config/meetings/mpc.csv`. `tests/data/boe/conditioning_paths.csv` is ground truth taken once from the Bank's Projections Databank.
- **`data/reference/`** is gitignored too: the full-sample outputs `scripts/regress.py freeze` wrote, one folder per currency, with a `meta.json` saying what code and which sessions they cover.
- **`reports/`** is generated: `build_panel.py` writes `coverage_USD.md`, `sofr_check_USD.md` and the figures above; `catalogue_vintages.py` writes `vintages_USD.md`; `build_nowcast.py` writes `nowcast_USD.md` and its figures; `build_model.py` writes `model_USD.md`, the model figures and the one-pager; `build_expression.py`, `build_strategy.py`, `build_portfolio.py`, `build_robustness.py` and `build_credit.py` write `expression.md`, `costs.md`, `portfolio.md`, `robustness.md` and `credit_USD.md`, their figures, and the numbers behind them as JSON in `reports/results/`, which later stages (the robustness grid's baseline check, the note) read instead of re-deriving.
- **`tests/data/`** is committed precisely so the suite runs for anyone who clones the repo without that archive. Everything in it is small enough to read in a diff. That includes a handful of single-day ZQ, SR1 and SR3 settlement strips from CME, taken from the licensed archive; everything added since week 7 is synthetic. To rebuild the full sample you need your own Databento licence for GLBX.MDP3. Regenerate with `uv run --env-file .env python tests/data/build_fixtures.py` (needs the archive and the network; `--market-only` leaves the ALFRED fixtures as they are).
- **`tests/data/fedwatch/`** holds hand-typed captures of the CME FedWatch tool. FedWatch publishes no history and it is not recoverable after the fact, so this gets filled in going forward rather than backfilled. Each capture stores the futures strip *and* the probabilities from the same screen, so comparing them isolates bootstrap-vs-bootstrap difference from data timing. Rows are laid out exactly as the tool displays them so a capture can be checked against the screenshot cell by cell, and every file opens with a one-line provenance note on line 1 (the readers skip it by position).

## Conventions

Five rules the code is written to and should keep being written to:

1. Every function that touches macro data takes an `as_of`. No "latest values" convenience overloads, no full-sample fits.
2. Network calls live only in `sources/`. Everything else reads from cache or committed fixtures. Observations carry both a reference date and a publication date.
3. Currency-specific behaviour lives in config, not in branches. Branching on the currency inside a module is a bug, and `tests/test_no_currency_branches.py` fails on one. Adding a currency should need only a config block (the schema in `config.py` says what is missing), a meetings file, and a source module if its data comes from a new provider.
4. The test suite passes on every commit, and `scripts/regress.py check` is identical after every refactor. A change that moves a number is a decision: it gets a row in `notes/DECISIONS.md` before the reference is refrozen.
5. Reports are generated, never hand-edited.
