# GlobalPolicyEngine

The finished project measures where markets misprice central-bank policy paths across currencies and trades the gap. What exists today is the first half of the USD leg: the **market** path, bootstrapped out of fed funds futures settlements and checked against two independent sources of ground truth. The model path, the signal and the backtest are not started.

## Status

Working end to end for USD:

- Daily ZQ (30-day fed funds) and SR3 settlements read out of a Databento GLBX.MDP3 archive on disk, one settle per `(trade_date, contract)`, each carrying both a reference date and a publication date.
- The FOMC meeting calendar, scraped and committed, covering 2020-01-29 through 2027-12-09 (65 meetings, including the two March 2020 inter-meeting cuts) with announcement date, effective date, and a `scheduled` flag. A piecewise-constant EFFR path solved between meeting effective dates from the whole futures strip at once.
- A test suite that validates the path against realized EFFR and against a captured CME FedWatch screen. `uv run pytest -q` gives 28 passed, 4 skipped.

## Install

Python >=3.11 (developed on 3.13). Dependencies are `pandas` and `databento`;
`requests` and `beautifulsoup4` are dev-only, used by the two scrapers that
never run inside the test suite.

```bash
uv sync
```

The package is src-layout (`src/policypath/`) and is installed editable by
`uv sync`, so it is imported as `policypath`, never off `sys.path`.

## Quick start

The test suite runs on committed fixtures and needs neither the network nor the
Databento archive:

```bash
uv run pytest -q
```

The demo scripts do need the archive (paid, gitignored -- see [Data](#data)):

```bash
uv run python scripts/run_implied_path.py 2022-06-01
```

```bash
uv run python scripts/show_databento.py 2024-03-15
```

## How the path is solved

`policypath.curves.policy_path.implied_path` takes the implied average rate per contract month (`100 - price`) and the meeting effective dates, and returns the overnight rate in force in each regime between them.

A ZQ contract settles to the arithmetic average of EFFR over every *calendar *day of its delivery month. So each contract month is one linear equation in the regime rates, with coefficients equal to the share of the month's days that each regime covers. The solver builds that weight matrix (months x regimes) and solves all months jointly by least squares.

Three consequences worth knowing:

- **The starting rate is an unknown**, backed out of the front contract, not supplied. Nothing in the solve is told what the current policy rate is, which is what makes the tests against published EFFR meaningful.
- **Joint least squares, not forward chaining.** Chaining month by month divides by the days remaining after an effective date, so a meeting on day 30 of 31 multiplies price noise ~15x and carries the error into every later meeting. On simulated 0.25bp price noise the median worst-meeting error was 0.8bp jointly versus 15bp chained.
- **It raises instead of guessing.** If the strip cannot identify the pillars (e.g. two meetings inside the only contract month), the weight matrix is rank-deficient and `implied_path` raises `ValueError` rather than returning a plausible-looking path.

Pillars are **effective** dates, one US business day after the announcement, because that is when the futures start averaging the new rate. CME, the Fed calendar and the press all label decisions by **announcement** date, so the two look off by one against each other. They are labels for different things; this was verified against all 20 published target-range changes since 2019.

## Data

- **`databento/`** is gitignored and not distributed: it is a paid batch archive, ~705 MB, one definition and one statistics file per day from 2020-12-31 to 2026-09-21 (1,795 days each). Only `sources/rates.py` reads it, and only the two demo scripts and `tests/data/build_fixtures.py` need it.
- **`tests/data/`** is committed precisely so the suite runs for anyone who clones the repo without that archive. Everything in it is small enough to read in a diff. Regenerate with `uv run python tests/data/build_fixtures.py` (needs the archive and the network).
- **`tests/data/fedwatch/`** holds hand-typed captures of the CME FedWatch tool. FedWatch publishes no history and it is not recoverable after the fact, so this gets filled in going forward rather than backfilled. Each capture stores the futures strip *and* the probabilities from the same screen, so comparing them isolates bootstrap-vs-bootstrap difference from data timing. Rows are laid out exactly as the tool displays them so a capture can be checked against the screenshot cell by cell, and every file opens with a one-line provenance note on line 1 (the readers skip it by position).

## Conventions

Five rules the code is written to and should keep being written to:

1. Every function that touches macro data takes an `as_of`. No "latest values" convenience overloads, no full-sample fits.
2. Network calls live only in `sources/`. Everything else reads from cache or committed fixtures. Observations carry both a reference date and a publication date.
3. Currency-specific behaviour lives in config, not in branches. Branching on the currency inside a module is a bug; adding a currency should need only a config block and a meetings file.
4. `tests/test_policy_path.py` passes on every commit.
5. Reports are generated, never hand-edited.
