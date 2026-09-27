# GlobalPolicyEngine

How well do markets price the path of central banks? This project is built to isolate these differences around currencies and trade them. 

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/implied_paths_USD_dark.png">
  <img alt="Implied fed funds path from ZQ futures on the first session of each month since 2010, against realized EFFR, and the implied move by meeting horizon as a heatmap" src="reports/figures/implied_paths_USD_light.png">
</picture>

## Status

Working end to end for USD:

- A cache under `data/` that one command rebuilds from nothing: ZQ, SR1 and SR3 settlements out of a Databento GLBX.MDP3 archive on disk, and EFFR, SOFR and the target range from FRED. Every observation carries a reference date and a publication date, and a revised value is a new vintage, not an overwrite.
- The FOMC meeting calendar, scraped and committed, with announcements from 2010-01-27 through 2027-12-08 (146 meetings). Each has an announcement date, an effective date, a `scheduled` flag and, for the 17-18 March 2020 meeting the Fed brought forward to the Sunday, the day it was called off. Each session sees the calendar as it stood that day: the two March 2020 emergency cuts are pillars only from their announcement, and the meeting they replaced is one until it was cancelled.
- A piecewise-constant EFFR path over the next eight meetings on every ZQ session from 2010-06-07 to 2026-09-21: 4,110 sessions, no solver failures, 99.68% of Fed business days. Of the 13 missing days, eight are Good Fridays, three are Fridays CME closed for a Saturday holiday, and two (2020-02-27, 2020-06-30) are days Databento flags as degraded, with no ZQ settle in the archive. The coverage report lists them.
- A SOFR discount curve bootstrapped in QuantLib from SR3 futures and SOFR fixings, and a cross-check of the ZQ path against SR1 (below).
- A real-time macro nowcast on every Fed business day from 2011-03-04, the first day every input has an ALFRED vintage (CBO's natural rate from 2011-02-02, average hourly earnings from 2011-03-04).
- A model path on each of the 3,921 sessions since then: the Fed's own balanced-approach rule from the Monetary Policy Report, in its inertial form, fed by that day's nowcast, with r* from the FOMC's longer-run dot and an explicit floor at the lower bound. The gap between the two paths, per meeting and z-scored on its trailing two years, and a crude backtest of it (below).
- `uv run pytest -q` gives 190 passed, 6 skipped.

## Install

Python >=3.11 (developed on 3.13). Dependencies are `pandas`, `pyarrow`, `databento`, `requests`, `pyyaml`, `matplotlib` and `QuantLib`, the last pinned to an exact version because its bindings change signatures between releases. `beautifulsoup4` is dev-only, used by the FOMC scraper, which never runs inside the test suite.

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

## Data

- **`databento/`** is gitignored and not distributed: it is a paid batch archive, ~1.2 GB, one definition and one statistics file per day per Databento batch job. Six jobs cover ZQ from 2010-06-06 to 2026-09-21, SR3 from 2020-12-31 to 2026-09-21, and SR1 from 2010-06-06 to 2026-09-24 (it first listed in 2018). Every job names its daily files the same way, so each is kept in its own folder, `databento/<definitions|statistics>/<job_id>/`, with its manifest, metadata and condition files. Download a new job straight into `databento/definitions/` or `databento/statistics/`, then run `uv run python scripts/file_databento.py`. It files every file under its job by the sha256 in the job's manifest, so the ` (2)` copies a browser makes of colliding names do not matter, and it checks every job is complete. `sources/archive.py` knows which job holds which root, and only `sources/` reads the archive.
- **`data/`** is gitignored and entirely derived. `data/cache/` holds one parquet vintage log per `(source, currency, series)` plus `manifest.json`, which records the ranges already covered and is what makes `update_data.py` incremental. CME sends a preliminary settle around 16:00 ET and the final that evening, or on the Sunday for a Friday session. Both are kept, so a read at the close sees the preliminary. The panel uses each session's settles as they stood by the end of the next business day, and records when the last one arrived.
- **`reports/`** is generated: `build_panel.py` writes `coverage_USD.md`, `sofr_check_USD.md` and the figures above; `catalogue_vintages.py` writes `vintages_USD.md`; `build_nowcast.py` writes `nowcast_USD.md` and its figures; `build_model.py` writes `model_USD.md`, the model figures and the one-pager.
- **`tests/data/`** is committed precisely so the suite runs for anyone who clones the repo without that archive. Everything in it is small enough to read in a diff. Regenerate with `uv run --env-file .env python tests/data/build_fixtures.py` (needs the archive and the network; `--market-only` leaves the ALFRED fixtures as they are).
- **`tests/data/fedwatch/`** holds hand-typed captures of the CME FedWatch tool. FedWatch publishes no history and it is not recoverable after the fact, so this gets filled in going forward rather than backfilled. Each capture stores the futures strip *and* the probabilities from the same screen, so comparing them isolates bootstrap-vs-bootstrap difference from data timing. Rows are laid out exactly as the tool displays them so a capture can be checked against the screenshot cell by cell, and every file opens with a one-line provenance note on line 1 (the readers skip it by position).

## Conventions

Five rules the code is written to and should keep being written to:

1. Every function that touches macro data takes an `as_of`. No "latest values" convenience overloads, no full-sample fits.
2. Network calls live only in `sources/`. Everything else reads from cache or committed fixtures. Observations carry both a reference date and a publication date.
3. Currency-specific behaviour lives in config, not in branches. Branching on the currency inside a module is a bug; adding a currency should need only a config block and a meetings file.
4. `tests/test_policy_path.py` passes on every commit.
5. Reports are generated, never hand-edited.
