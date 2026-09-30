# Build spec v2: Weeks 7–13 of the Global Policy Path Engine

## Status at the 2026-09-29 checkpoint (read this first when resuming)

- **Done, reviewed and fixed:** Phase 0 (§2: data layer, config, `config/strategy.yml`, schema); week 7 (§3: `signal/components.py`, `strategy/{instruments,carry,expression}.py`, `report/expression.py`, `scripts/build_expression.py`, the brief's trades table); week 9 model variants (§5.1: override plumbing, HLW, `model/estimate.py`, `curves/nss.py`, converge-to-target); week 12 (§8: `regimes.py`, `credit.py`, `report/credit.py`, `scripts/build_credit.py`, `notes/credit_section.md`).
- **Verified at the checkpoint:** `uv run pytest -q` passes 812 (6 skipped); `scripts/regress.py check` is IDENTICAL for USD and GBP; `build_expression.py` and `build_credit.py` rebuild their reports byte for byte; the brief fits one page.
- **Week 8 done 2026-09-29** (§4: `strategy/{risk,positions,costs}.py`, `scripts/build_strategy.py`, `reports/costs.md`; agent-verified: 861 passed, regress IDENTICAL).
- **Week 9 done 2026-09-29: portfolio and robustness** (§5.2–5.3: `strategy/{portfolio,robustness}.py`, `report/{portfolio,robustness}.py`, `scripts/build_{portfolio,robustness}.py`, `reports/{portfolio,robustness}.md`; reviewed, fixed, mutation-tested; every harness mutant is killed). Integration pass: `build_portfolio` (about 4s) then `build_robustness` (about 10s, reads the portfolio code and stops unless its baseline equals `reports/results/portfolio.json`'s headline) each rebuild their reports, JSON and figures byte for byte; `uv run pytest -q` passes 944 (6 skipped); `scripts/regress.py check` is IDENTICAL for USD and GBP. Headline (inverse-vol, shrunk): gross -0.15 (0.29), net -0.62 (0.32) over 2,970 sessions. **Next, in order (§12):** week 11 (§7, the note, which includes `notes/credit_section.md` from its `## Credit` heading), and week 13 (§9).
- **Week 10 done 2026-09-29, on the Windows machine** (§6: `backtest/metrics.py` completed, `report/attribution.py`, `report/tearsheet.py`, `report/charts.py` completed, `scripts/build_tearsheet.py` → `reports/tearsheet_<date>.pdf`, `reports/metrics.md`, `reports/results/metrics.json`, `reports/figures/attribution_{light,dark}.png`; DECISIONS A1–A11). One command, about 10s; md and JSON rebuild byte for byte. The level carries 63% of the book's gross P&L variance (the report says the level dominates); ELB flat -0.62, exclude -0.56, hold -0.70; ex-2022 -0.99 (0.37), ex-2022-23 -1.13 (0.40). `uv run pytest -q`: 956 passed, 6 skipped, and 6 bit-for-bit regression checks fail on Windows only (frozen on the Mac, last-bit float differences; A11, not refrozen). Week 11 reads `reports/results/metrics.json`.
- **After an outside review, 2026-09-29 (Windows): report additions, no model or strategy change** (DECISIONS A12-A17). Added: the cost reconciliation (verified by hand, and correct), carry against rate by sleeve, the break-even information, the carry filter at entry, and the level the z removes. The tear sheet gains IC(21), and the brief shows the differential in z only. Two corrections to the review, for week 11: (1) right and bleeding holds for GBP outright and USD 2s10s, not the book, because the USD outright loses on its rate calls; (2) the USD discount to the rule is persistent (13 of 13 years) but not stable (-25 to -86bp), and GBP's changes sign. Week 11 should build the note on these, not on the review's framing.
- **Build order for the outputs:** `build_panel` → `build_nowcast` → `build_model` (per currency) → `build_expression` → `build_credit` → `build_brief`. From the panels and the cache, independently of those: `build_strategy`, and `build_portfolio` → `build_robustness` (the grid checks its baseline against `reports/results/portfolio.json`).
- **Open author items:** `notes/author_review.md`. Proposed decisions: C18 (the breakeven's verdict turns on 2022), V4/V9 (the estimated-rule sample), D12–D15, D17–D19 (credit readings; the predictive result is at the edge of chance, placebo one-sided p = 0.07).
- **Runtime lesson:** keep foreground commands under about 150s and tests synthetic; run `regress.py check --ccy USD` and `--ccy GBP` separately; at most two heavy chains in parallel. macOS has no `timeout` command.
- The scratchpad paths below belong to the first session and may be gone: this file and `notes/design_review_w7_w13.txt` are the durable copies.

Repo: `/Users/ericbeechen/Projects/Cobalt/GlobalPolicyEngine` (package `src/policypath`; run everything with `uv run` from the repo root).
Plan: `/Users/ericbeechen/Desktop/PLAN_W7_W13.md` (read §0 and your week in full before starting).
Scratchpad (review artefacts, downloaded workbooks): `/private/tmp/claude-501/-Users-ericbeechen-Projects-Cobalt-GlobalPolicyEngine/31471aba-6a7f-487d-a4f9-63a701360377/scratchpad/`.
This spec turns the plan into concrete designs, after an adversarial design review (`scratchpad/review.txt` holds its findings, with the data evidence). Where the spec and the plan disagree, the spec wins; §11 says why.

---------------------------------------------------------------------------------------------------

## 0. Ground rules for every agent

1. **Standing invariants (plan §0) are hard constraints.**
   - Every function touching macro data takes an `as_of` date. No "latest" convenience.
   - Network calls live only in `src/policypath/sources/`. Everything else reads the cache (`sources/cache.py`) or `data/panel/`.
   - Currency-specific values live in `config/currencies.yml` (per currency) or `config/strategy.yml` (book level). **No currency codes, rate names, contract roots, committees or series codes in code** outside the backend modules listed in `tests/test_no_currency_branches.py` (`BACKENDS`). New shared modules must pass that test (Phase 0 extends its regex with the new series codes). Docstrings/comments may name currencies as examples. A module that is one currency's by nature (the credit bridge) iterates over the enabled currencies whose config has the block, and names outputs `..._{ccy}`.
   - `tests/test_policy_path.py` and `tests/test_no_lookahead.py` pass after your change.
   - Reports (`reports/`) are generated by scripts, never hand-edited.
   - Never write raw vendor data (CME settles, ICE, Moody's) to a tracked file. `data/` is gitignored. New tests use **synthetic** data only. (The repo already tracks some raw CME strips under `tests/data/`; that is a known finding for week 13 to report, not for anyone to change.)
2. **No new dependencies.** Available: pandas, numpy, matplotlib, pyarrow, pyyaml, openpyxl, requests, quantlib, databento; pytest. No scipy/statsmodels/sklearn: write OLS, Newey–West, shrinkage, NSS in numpy.
3. **Bit-identical baseline.** Any change to an existing module the chain runs (market, panel, curves, model, macro, signal, backtest, regress, config loading) must leave `uv run python scripts/regress.py check` (full tier, both currencies, ~60s) and `uv run pytest tests/test_regression.py` IDENTICAL. Never refreeze; never add a stage to `regress.STAGES` (the fixture test asserts the list). If you believe a refreeze is needed, stop and report why.
4. **Every new module gets a test** (offline, fast, synthetic). Whole suite is ~7s today: keep new tests to a few seconds total.
5. **Config schema changes update the validation** in `src/policypath/config.py` (currency blocks: `SCHEMA` and the conditional tables; book config: `STRATEGY_SCHEMA` checked by `config.strategy()`), with tests in `tests/test_config.py` (extend its `required()` enumeration).
6. **Decisions.** Every non-obvious parameter choice gets a row in `notes/DECISIONS.md` (what, date 2026-09-28, why, where; which numbers moved, if any). Append rows to your week's section (§10), with the `Edit` tool, never overwrite the file. Judgements the plan reserves for the author (ELB choice, headline construction, the honest reaction to post-cost results, the note, the verbal version) are marked **proposed** in their row.
7. **Author checklist.** Append one line per author-owned artefact or proposed decision to `notes/author_review.md` (create it if missing): what, where, the DECISIONS ID, and the config key that flips it.
8. **Style: match the repo.** Read two or three existing modules first. House style: a module docstring saying what the module is and *why* each choice was made, with the numbers; short functions; docstrings with units and sign conventions; no comment noise, no dead code, no commented-out blocks. Rates in percent in panels; bp for gaps and P&L. Report prose: short declarative sentences, numbers inline, no hype.
9. **Do not commit, push, or touch git history.** Leave changes in the working tree.
10. **Do not run** `scripts/update_data.py` (except Phase 0, only with `--only`), `scripts/build_panel.py`, `scripts/build_nowcast.py`, `scripts/build_model.py`. The panels in `data/panel/` are current. Scripts you add write only their own outputs.
11. **Concurrency.** Some phases run in parallel (§12). Edit only the files your phase owns (§12). If a test fails in a file another phase owns, do not edit it: re-run later, and report it if it persists.
12. `.env` holds `FRED_API_KEY`; fetch scripts run as `uv run --env-file .env python ...`.

Existing APIs to reuse (read them): `sources/cache.py` (`log`, `read`, `view`, `Presorted`), `market.py` (`extractor`, `settles_on`, `monthly_strip`, `_as_published`, `ForwardCurveExtractor._curve`, `rate_in_force`), `curves/forward.py` (`forward_path`), `calendars.py` (`BDAYS`, `known_daily`, `known_meetings`), `panel.load(ccy, name)`, `signal/gap.py` (`zscore`), `signal/cross.py` (`differential`), `backtest/engine.py` (`run`, `stats`), `backtest/instrument.py` (`held_rate_change`), `report/figures.py` (`THEMES`, `_style`, `_title`), `report/onepager.py` (`_Column`, `PAGE`; `_check` raises past one page), `report/brief.py` (`tags`, `footer`), `regress.py` (`merge`, `outputs`).

Panels in `data/panel/` (both currencies): `<ccy>_sessions` (session, error, last_fixing, rate_now, …; GBP also rate_in_force), `<ccy>_meetings` (session, k, announcement_date, effective_date, rate, step_bp, cum_bp), `<ccy>_macro`, `<ccy>_model` (session, inflation, u_gap, rstar, notional, at_elb, elb, goal, r0, spread_bp, …), `<ccy>_paths` (session, k, effective_date, market, model), `<ccy>_signal` (session, k, gap_bp, z, window_mean_bp, window_sd_bp), `<ccy>_backtest` (session, position, change, pnl, equity). USD also `USD_sofr_basis`.

---------------------------------------------------------------------------------------------------

## 1. Conventions (all weeks)

### 1.1 Units and signs
- **Position `q`** of a leg: DV01 in the **book currency** (`strategy.yml: book.currency`, USD) per bp. **Positive = receive** (long the price; gains when the rate falls), as `backtest/engine.py`.
- A **unit position** is `q = +1`; unit P&L is in **bp**; dollars = unit P&L × q.
- Rates in percent (as panels store them); P&L components converted to bp (× 100) where computed.
- **Direction:** `z > 0` ⇒ receive the sleeve's first leg. For every sleeve `z > 0` means the market prices *more* tightening than its reference (the rule, or the other currency), so the priced rate is "too high".

### 1.2 Sessions, timing and marks
- A sleeve has a session calendar (its signal's sessions). On session `t`: the signal `z_t` is known at `t`'s close (GBP's curve by noon `t+1`, as the path panel treats it); the **instrument selected** at `t` is what a trade at `t`'s close puts on; the **position decided** at `t` executes `execution_lag` sessions later (currency `backtest.execution_lag`, 1) at that session's close, in the instrument selected at the execution session.
- **P&L credited to `t`** is earned by the position held from the previous sleeve session `p` into `t`, in the instrument selected at `p` ("held"). Matches `engine.run` (decided `lag+1` sessions earlier) and `instrument.held_rate_change`.
- `Δ` = calendar days from `p` to `t`.
- **Marks.** A leg's curve on day `d`, `y_d(·)`, is the print **dated `d`**, as it stood at the end of `d + lag` business days of its source (reuse `market._as_published`; build a by-date wide frame once, never per-session `cache.view`). If `d` has no print (e.g. Columbus/Veterans Day: CME open, bond market shut; GLC holiday rows), carry the last print dated on or before `d` forward: its rate change is 0 and carry still accrues. Count stale sessions per leg and report them. Never read `cache.read(as_of=t)` for a mark: that returns the print dated `t−1` and books the move between decision and execution.
- **Instrument ids** are unique in time: a ZQ contract is its **expiration month** (monthly_strip's PeriodIndex), never its symbol (ZQF1 is both Jan 2011 and Jan 2021). A forward is its window `(E_k, E_{k+1})`. A par leg is its tenor.

### 1.3 Attribution: identity, revaluation, convergence
For a unit receive held from `p` into `t` in instrument `h`, with `τ_d(h)` its maturity (or time-to-start) as of `d`:
```
held_p = y_p(τ_p(h))    held_t = y_t(τ_t(h))    rolled = y_p(τ_t(h))    fx = S_t / S_p (1 for book-currency legs)
carry  = +100 × ( y_p(T) × Δ/365  −  r_p × Δ/basis ) / A_p × fx     funded (spot-starting par) legs only, else 0
roll   = −(rolled − held_p) × 100 × fx
rate   = −(held_t − rolled) × 100 × fx
```
`r_p` = the overnight rate in force at `p` (`<ccy>_sessions.rate_now`; GBP `rate_in_force` if present), `basis` = the currency's `overnight.day_count` (USD 360, GBP 365), `A_p` = modified duration of the par instrument (years), `T` the constant maturity. Funding at the overnight rate stands in for GC/gilt repo: tag it.

Three checks, all in `strategy/carry.py`, all run on **every sleeve, both directions, every session** in the build script and on synthetic data in tests:
1. **Bookkeeping identity (exact):** `linear_total = [−(held_t − held_p) × 100 + carry_unfx] × fx` equals `carry + roll + rate` to 1e-9 bp. This catches a relative roll/rate sign error and maturity bookkeeping; it is nearly tautological by construction and **does not prove the signs**. Say so in the docstring.
2. **Independent full revaluation (tolerance from convexity):** total P&L recomputed from the instrument's own valuation, not from the rate-space pieces. ZQ: `contracts × (P_t − P_p) × point_value`, `contracts = q / dv01_contract`, `point_value = dv01_contract × 100` per 1.00 of price, settle prices as published. Forward: `N × α × (K − F_t) × DF_t(E_{k+1})` for a receiver struck at `K = F_p` (value 0 at `p`), `N` from the leg's DV01 at `p` (`α × DF_p(E_{k+1}) × 1e-4`), FX at `S_t`. Par: a bond struck at `p` with coupon `c = y_p(T)`, priced with the standard semiannual bond formula at `y_t(T − Δ/365)` (clean, no accrued-interest bookkeeping needed for a par strike), plus coupon accrual `c × Δ/365` minus funding `r_p × Δ/basis`, on notional `N = 1e4/A_p`, × fx. Residual `full − (carry + roll + rate)` must be within `0.5 × C × (Δy)² × N` (+ a small cross-term allowance for the forward's ΔF×ΔDF and the fx cross term) — report its max and the bound. A flipped carry sign shows up here as ~2× carry, far above the bound. (Corrected at the week-7 review: an FX quote inverted *at its source* moves both sides together, so this check cannot see it; the guard is the config range `expression.fx.plausible`, and the build refuses a spot outside it.)
3. **Convergence (pins the direction):** synthetic test per sleeve: move the market to the model between `p` and `t` (everything else frozen). Every sleeve with `z > 0` must gain; a fixed-window leg must gain exactly the (de-meaned) gap. Plus the end-to-end check in the build script: the **GBP outright sleeve with the linear rule and fx forced to 1 reproduces `data/panel/GBP_backtest.parquet` pnl on every session from its first z** (the held k=4 forward equals the path's held-meeting rate for k ≥ 2), to 1e-9.

`tests/test_carry.py` also pins each of the **12 sign cases** (3 expressions × 2 currencies × 2 directions) with hand-checkable worked numbers, e.g. receive on an upward curve: positive roll, positive carry when `y > r`; receive on an inverted curve: negative roll and carry ("right and bleeding"); pay on an inverted curve: positive carry and roll; long 1 ZQ = +$41.67 per bp fall; receive-GBP/pay-USD 2y gains when GBP 2y falls relative to USD 2y and its GBP P&L scales with `S_t`.

Convexity is not modelled in the linear P&L: legs are DV01-linear marks re-struck at each close. For a DV01-neutral 2s10s it is ~3.6bp/yr per unit DV01 (reviewer's estimate): state it in the docstring and a DECISIONS row, and report the realized full-reval residual sum per year as the measured size.

### 1.4 FX
GBP legs convert at spot: `S_t` = USD per GBP (FRED `DEXUSUK`, H.10 noon NY), last print on or before `t` (§1.2 rule; lag 0). A unit (+1 USD DV01) GBP leg holds GBP DV01 `1/S_p` from `p`; its GBP P&L converts at `S_t`: `fx = S_t/S_p` on every component. Config-driven: the GBP block's `expression.fx` names the series; the USD block has none because USD is `book.currency`.

### 1.5 Signal components (`signal/components.py`)
- **level** (outright sleeves): gap and z at `k = backtest.horizon` (4) from `<ccy>_signal`.
- **slope** (curve sleeves): `slope_bp = gap_bp(k = slope[1]) − gap_bp(k = slope[0])` (8th minus 1st). On the current panels it is 85–94% the level (corr of z 0.85 USD, 0.94 GBP), so the traded slope is **orthogonalised**: `slope_orth_t = slope_bp_t − β_t × level_bp_t`, `β_t` the OLS slope of slope_bp on level_bp over the trailing `signal.window` strictly before `t` (`closed="left"`, `min_periods` as the signal spec), then z-scored with `signal.gap.zscore` and the currency's `signal` spec. Two sentences: "the slope component is the gap at the eighth meeting minus the gap at the first, less the part the level gap explains over the trailing two years." The raw 8-minus-1 z is kept alongside and is a robustness row.
- **differential** (cross sleeve): `signal.cross.differential(first, second, spec)` at `k = backtest.horizon` (both currencies' horizon must be equal: check and raise).
- Every component carries `value_bp`, `z`, `window_mean_bp`, `window_sd_bp` and the **de-meaned edge** `dev_bp = value_bp − window_mean_bp` (= z × window_sd_bp).

### 1.6 The edge and the breakeven (the week-7 insight)
- **Edge** of a signal = `side × dev_bp` = `|z| × window_sd_bp` ≥ 0: the bp the signal expects to close (the move to its trailing mean, where the hysteresis exit sits). **Never the raw gap**: the raw gap carries the constant offsets (r*, u*, spread) that z removes, and its sign disagrees with z on 34% of USD |z|≥1 signals. The raw gap is shown as the quoted number only.
- For a ZQ leg whose contract month straddles meeting k+1 (44.5% of USD sessions), the edge is the day-weighted average of the de-meaned gaps of the regimes the month covers (from the signal panel, k and k+1).
- **Carry and roll ahead** `CR_h` = carry + roll over `h = carry.horizon_days` (91) for the instrument selected at the session, held `h` days, if the constant-maturity curve does not move (roll `= −(y(τ−h) − y(τ)) × 100`; carry as §1.3 over `h`), signed for the side. This is for a fixed instrument held h days, not for the sleeve's re-selection: say so. For a forward window whose rolled start falls below the curve's first node (1m), read the rate in force there (as `pin_first_regime` does).
- **Right but bleeding**: `CR_h < 0 < edge`.
- **Does it pay for the bleed?** Over `h`, let the held rate close a fraction `φ` of the edge and otherwise follow the frozen curve: `E_h = φ × edge + (1 − φ) × CR_h` (this is exact for a fixed-window instrument: the roll cancels under full convergence, so there is no double counting). The break-even closure is `φ* = −CR_h / (edge − CR_h)` when `CR_h < 0` (else 0). The empirical closure `φ_h` is estimated **expanding-window** per sleeve: OLS without intercept of `−(value_bp_{t+h} − value_bp_t)` on `dev_bp_t` over pairs with `t + h ≤ as_of` (h in sessions ≈ the calendar horizon), clipped to [0, 1], NaN until 250 pairs. A signal **does not pay for its bleed** when `φ* > φ_h` (equivalently `E_h < 0`). Two sentences for the note: "Over a quarter the gap has historically closed by a fraction φ; the rest of the time the curve stands still and the trade earns its carry and roll. If that expected quarter is negative, the signal is right and still not worth trading."
- For the curve and cross sleeves the component's bp is a *proxy* for the instrument's move (slope gap for 2s10s, the k=4 differential for the 2y spread): say so wherever it is reported.
- `positions.carry_filter` (skip entries that do not pay for their bleed) is a **diagnostic**, default off; its effect is reported, and it is a robustness row.

---------------------------------------------------------------------------------------------------

## 2. Phase 0: data layer and config

### 2.1 New cached series

| Block | Source | Series | Publication lag | Use |
|---|---|---|---|---|
| USD `sources.daily.fred` | FRED | `DGS1 DGS2 DGS3 DGS5 DGS7 DGS10 DGS30` | 1 (H.15 next business day) | curve and cross legs (CMT par yields, semiannual); DGS30 for the credit maturity control |
| USD `sources.daily.fred` | FRED | `BAA10Y AAA10Y` | 1 | credit bridge (Moody's seasoned Baa/Aaa minus 10y CMT) |
| USD `sources.daily.fred` | FRED | `BAMLC0A0CM BAMLH0A0HYM2` | 1 | credit cross-check only (ICE BofA OAS: FRED carries only 2023-09-29 onward) |
| GBP `sources.daily.fred` | FRED | `DEXUSUK` | 0 (a market price seen on the day; FRED's H.10 copy arrives weekly, but the rate is not a statistic) | FX |
| GBP `sources.curves.boe` | BoE | `GLC_SPOT` (new) | 1 | GBP par legs: the Bank's fitted nominal gilt **spot** curve, sheet "spot curve" with first maturity 0.5y (half-yearly to 25y, 40y from 2016), continuously compounded, from `glcnominalddata.zip` + `latest-yield-curve-data.zip`, keyed (date, tenor months) like `OIS_SPOT` |
| USD `sources.estimates.nyfed` (new kind) | NY Fed | `HLW_RSTAR` | vintage quarter end + `lag_days` (62) | robustness r*: the HLW **real-time** vintages file `Holston_Laubach_Williams_real_time_estimates.xlsx` (linked from newyorkfed.org/research/policy/rstar): one sheet per vintage, 2015Q4–2020Q2 (HLW 2017 model) and 2022Q4–2026Q2 (HLW 2023 model), no vintages 2020Q3–2022Q3. Value per vintage = the vintage's final-quarter one-sided US r* (column under the "Natural Rate (r*)" group, country header from the backend or config). `date` = the vintage's final quarter start; `published` = that quarter's end + `lag_days` (release dates run 58–62 days after quarter end). The model holds a vintage until the next, so the 2020Q2 vintage stands through the gap. |

Implementation:
- `sources/boe.py`: generalise the parser to choose the monthly short-end spot sheet (first maturity 1/12: existing behaviour, **output unchanged**) or the half-yearly long-end spot sheet (first maturity 0.5, and 'spot' in the sheet name). Give every `CURVE_FILES` entry the same arity (e.g. `(archives, prefix, kind)`); cache parsed **archives** on the instance so `latest-yield-curve-data.zip` downloads once per update, not once per series; skip workbooks wholly before `start` by the filename's year range. The GLC long-end sheet lacks the 0.5y node on 20–48% of days: store what is there (NaN dropped, as today); `par_from_spot` handles it (§3).
- `sources/nyfed.py` (new backend; add to `BACKENDS` with the reason): `Hlw(Source)`; parse each vintage sheet; tests on a synthetic two-layout workbook.
- `sources/registry.py`: `ESTIMATES = {"nyfed": _nyfed}` with `_nyfed(lag_days)`; config shape `sources: {estimates: {nyfed: {HLW_RSTAR: 62}}}` validated by a `_lags`-style checker (optional path); `scripts/update_data.py` loops over it like the daily loop.
- `scripts/update_data.py`: `--only SERIES [...]` (fetch only these; skip everything else including macro vintages). Fix the docstring: GBP updates now need the FRED key (DEXUSUK).
- `config.py`: `_cached` covers `estimates`; `_refs` adds every series the new blocks read (expression par series, fx, credit series). A tenor-map validator for `{tenor: series}`. Conditional tables for the new optional blocks (`expression` by instrument type; `fx` required for a currency that is not `book.currency`; `credit` optional; `contracts` required when an instrument names one). Validate every `tags` block's shape (`{name: {kind: text}}`, kind ∈ {lag, quality, modelling}). `config.strategy()` loads and validates `config/strategy.yml`, and cross-checks it against the currency blocks (sleeve currencies enabled; the expression each sleeve needs exists; equal horizons for a cross pair) — in `strategy()`, not in `_currencies()` (whose temp-dir test has no strategy.yml).
- `tests/test_no_currency_branches.py`: extend `FORBIDDEN` with `DGS\d+|DEXUSUK|BAA10Y|AAA10Y|BAML\w+|GLC_SPOT|OIS_SPOT|HLW_RSTAR` (current shared code must still pass; if `OIS_SPOT` appears outside a backend, move it to config).
- Config comment fix (no output change): USD `rule.spread_breaks: []` says "EFFR's definition did not change inside the sample"; the NY Fed changed EFFR's methodology on 2016-03-01 (brokered mean → FR 2420 volume-weighted median). Correct the comment; add a DECISIONS row (R10) saying why it is not a break: the 20-fixing median is past it within a month and the spread moved by under a basis point (check `EFFR − midpoint` median in the 20 fixings either side and quote it).
- Fetch: `uv run --env-file .env python scripts/update_data.py --ccy USD --only DGS1 DGS2 DGS3 DGS5 DGS7 DGS10 DGS30 BAA10Y AAA10Y BAMLC0A0CM BAMLH0A0HYM2 HLW_RSTAR` and `--ccy GBP --only DEXUSUK GLC_SPOT`. Report first/last date, rows, and gaps per series.
- Tests (offline, synthetic workbooks): the half-yearly parser; the monthly parser's output unchanged on a synthetic monthly sheet; HLW parser (two layouts, a gap); `published` rules; registry lookups; config refs, tenor maps, tags shape, strategy cross-checks.
- After: `uv run pytest` green; `regress.py check` IDENTICAL for both currencies.

### 2.2 Config additions

Each currency block, after `backtest:` (GBP values in comments):
```yaml
  # Week 7: what each expression trades in this currency (strategy/instruments.py names the types).
  expression:
    outright:
      instrument: futures_month      # GBP: curve_forward (a meeting-dated OIS forward off market.curve)
      contract: ZQ                   # the key under contracts (futures_month only)
    curve:
      instrument: par_yield          # GBP: par_from_spot
      source: fred                   # GBP: boe
      series: {1: DGS1, 2: DGS2, 3: DGS3, 5: DGS5, 7: DGS7, 10: DGS10}   # GBP: GLC_SPOT (one curve)
      legs: [2, 10]                  # receive the first, pay the second, DV01-neutral
      coupons_per_year: 2
    fx: {source: fred, series: DEXUSUK, book_per_unit: true}             # GBP only
    tags: {...}
  contracts:                         # USD only. DV01 is derived from these, never typed.
    ZQ:
      notional: 5000000
      accrual: {days: 30, year_days: 360}   # DV01 = 5,000,000 x 30/360 x 1bp = $41.6667 (CME rounds to $41.67)
      tick: {front: 0.0025, other: 0.005}   # $10.4167 and $20.8333 derived (CME quotes $10.4175, $20.835)
      fee_per_side: 1.0                     # USD a contract, exchange and clearing, assumed
  costs:                             # Week 8: round trip per leg, bp of yield unless in ticks
    outright: {ticks_round_trip: 2, observed: true}      # USD: two back-month ticks, 1bp; GBP: {round_trip_bp: 1.0, observed: false}
    curve: {round_trip_bp: 0.5, observed: false, roll_every_months: 3}   # GBP 1.0
    tags: {...}
```
and `overnight.day_count: 360` (USD) / `365` (GBP) inside the existing `overnight` block.
USD also:
```yaml
  credit:
    spreads:
      quality: {long: {source: fred, series: BAA10Y}, short: {source: fred, series: AAA10Y}}   # Baa - Aaa: maturity-matched, primary
      baa_30y: {long: {source: fred, series: BAA10Y}, plus: {source: fred, series: DGS10}, minus: {source: fred, series: DGS30}}  # DBAA - DGS30
      baa_10y: {long: {source: fred, series: BAA10Y}}
    control: {long: {source: fred, series: DGS30}, short: {source: fred, series: DGS10}}      # the Treasury 10s30s
    crosscheck: {ig_oas: {source: fred, series: BAMLC0A0CM}, hy_oas: {source: fred, series: BAMLH0A0HYM2}}
    tags: {...}
```
(The exact credit shape is the credit agent's to finalise: keep it declarative, validated, series named only here.)

Remove nothing yet; week 8 replaces the stale `backtest.tags.costs` ("no transaction costs yet (week 8)") with the real cost tags.

New **`config/strategy.yml`** (validated by `config.strategy()`):
```yaml
book:
  currency: USD
  capital: 100000000
  vol_target: 0.05
sleeves:
  - {name: USD outright, kind: outright, ccy: USD}
  - {name: GBP outright, kind: outright, ccy: GBP}
  - {name: USD 2s10s, kind: curve, ccy: USD}
  - {name: GBP 2s10s, kind: curve, ccy: GBP}
  - {name: GBP - USD 2y, kind: cross, pair: [GBP, USD], tenor: 2}   # receive the first's 2y, pay the second's, DV01-matched
signal:
  slope: [1, 8]
  slope_orthogonal: true
carry:
  horizon_days: 91
  closure_min_pairs: 250
positions:
  rule: hysteresis
  enter: 1.0
  exit: 0.0
  grid: {enter: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], exit: [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]}
  carry_filter: false
costs:
  sensitivity_bp: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
risk:
  ewma_lambda: 0.97
  sigma_floor: 0.5            # x the trailing two-year median sigma
  nonsynchronous_lag: 1
  no_trade_band: 0.10
  max_gross_dv01_per_capital: null   # set by the portfolio agent with a DECISIONS row
portfolio:
  constructions: [erc, mean_variance, inverse_vol]
  headline: inverse_vol       # proposed
  z_cap: 3.0
  min_history: 250
  drawdown: {trigger: 0.10, release: 0.05, scale: 0.5}
evaluation:
  elb: {chosen: flat}          # proposed; flat | exclude | hold
  ic_horizons: [5, 21, 63]
  exclude: {ex_2022: ["2022-01-01", "2022-12-31"], ex_2022_23: ["2021-12-01", "2023-08-31"]}
robustness: {}
tags: {}
```
Values are decisions: each gets its DECISIONS row in the week that uses it.

---------------------------------------------------------------------------------------------------

## 3. Week 7: expression layer, carry and roll

### Modules
- `signal/components.py` (§1.5).
- `strategy/instruments.py` — legs, chosen by the config `instrument:` name through a registry. Each leg, built once for a list of sessions from wide by-date frames (§1.2), gives per session: instrument id, its rate on that session, its rate on the previous sleeve session's data at the rolled maturity, native DV01 per unit notional, funding and duration (funded legs), FX, and the full-revaluation inputs (§1.3). Types:
  - `futures_month`: the contract = the first calendar month starting on or after the k-th meeting's effective date (from `<ccy>_meetings` at the session). `y = 100 − P`, settles via `market.settles_on(..., after=market.final_after_bdays)`. Maturity coordinate: days to the month's midpoint; the day's strip is linear in it between months, flat beyond. `dv01_contract = notional × accrual.days / accrual.year_days × 1e-4`; tick values derived. Carry 0.
  - `curve_forward`: window `[E_k, E_{k+1})` (next effective date from the meetings panel, or `E_k + path.tail_days` at `k = n_meetings`); the forward from the day's `market.curve` with log-DF linear between nodes (factor a helper out of `curves/forward.py` **without changing `forward_path`'s bits**). For `k ≥ 2` equals the path's `rate` at k to 1e-10 (test). Rolled window `[E_k − Δ, E_{k+1} − Δ)` on `p`'s curve; below the first node read the rate in force. Native DV01 `α × DF(E_{k+1}) × 1e-4`. Carry 0.
  - `par_yield`: CMT par yields by tenor; `y_d(T)` linear in maturity between tenors; a leg struck at `p` has maturity `T − Δ/365` at `t`; `A = [1 − (1 + y/f)^(−fT)]/y`, `A = T` when `|y| < 1e-10`; native DV01 `A × 1e-4`.
  - `par_from_spot`: `DF(t) = exp(−s(t) t)` with `s·t` linear in t between the day's nodes and 0 at t = 0 (so a missing 0.5y node is interpolated, never a crash, never a method switch); `c(T) = f (1 − DF(T)) / Σ_{i=1..fT} DF(i/f)`; then as `par_yield` (par yields at the rolled maturity by linear interpolation between half-year grid points). Test with the 0.5y node missing.
- `strategy/expression.py` — sleeves: signal → side → legs → positions (book DV01) → per-session unit P&L by component and leg. Outright: one leg `q = s`. Curve: receive `legs[0]`, pay `legs[1]`, `|q| = s` each (DV01-neutral). Cross: receive the first currency's `tenor`-year par leg, pay the second's, `|q| = s` in book DV01 (matched at spot), on the sessions both have. Rules: `linear` (`s = z`, week 6's sizing) now; `hysteresis` and ELB treatment in week 8 (`strategy/positions.py`).
- `strategy/carry.py` (author-owned draft): every carry/roll/funding formula lives here (instruments.py supplies rates and maturities only): `attribution` (§1.3 components, the three checks), `carry_roll_ahead` (§1.6), `edge`, `closure` (the expanding φ_h), `bleed` (right-but-bleeding, φ*, E_h, pays?). Docstrings state every formula and sign so the author can re-derive them.
- `report/expression.py` + `scripts/build_expression.py` → `data/panel/book_expression.parquet`, `reports/expression.md`, figure(s) `reports/figures/carry_{light,dark}.png`, `reports/results/expression.json` (headline numbers for the note), `reports/results/signals.csv` (derived, one row per signal episode: entry date, sleeve, z, edge, raw gap, carry, roll, CR_h, φ*, φ_h, pays?, subsequent P&L over h). Report (generated):
  - instruments with the ZQ DV01 derivation printed from specs and tick values (derived vs CME's rounded quotes);
  - the three checks: sessions checked per sleeve and direction, max |identity residual|, max full-reval residual vs its bound, the GBP end-to-end reproduction, stale-mark counts per leg;
  - P&L decomposition (carry, roll, rate, total) per sleeve, linear rule, pre-cost, by year;
  - **the insight**: over signals (sessions with |z| ≥ `positions.enter`, and all sessions), the share right-but-bleeding, the share that do not pay for their bleed, their subsequent P&L against the others, and the carry filter's effect (diagnostic);
  - corr(level z, raw slope z), corr(level z, orthogonal slope z), corr(curve sleeve P&L, outright P&L) per currency;
  - the current signal table: per sleeve, the latest session, z, side, instrument (e.g. "ZQ Apr 2027", "2y gilt vs 2y Treasury"), raw gap, edge, carry and roll over h, CR_h, pays?, and the position in contracts/notional for a $10,000-per-bp unit.
- **Brief**: add a compact "Trades" table (per sleeve: z, side, instrument, breakeven CR_h, right-but-bleeding / pays?). Make room: the footer reads tags only from the blocks the brief itself reads (list them in `config/brief.yml` as `footer_blocks`; credit, HLW, robustness tags go to the generated limitations only); chart height 2.7in → ~2.2in; the table at 7.5pt. Regenerate `reports/brief_2026-09-28.pdf` with `scripts/build_brief.py --date 2026-09-28 --preview <scratch png>` and look at the PNG.

### Acceptance (plan)
Identity holds across all expressions and dates (reported, with the stronger checks); carry, roll and breakeven per signal; ZQ DV01 from contract specs (a code-token test, like `code_hits`, asserts 41.67/41.6667 appears in no code token); brief regenerated. No costs, no portfolio, no optimisation.

---------------------------------------------------------------------------------------------------

## 4. Week 8: costs, turnover, hysteresis, ELB treatment

- `strategy/risk.py`: the EWMA machinery shared by weeks 8–9: `ewma_sigma` (λ, strictly before t), `sigma_floor` (max(EWMA σ, floor × trailing 2-year median σ)), `ewma_cov` (below, week 9).
- `strategy/positions.py`: `linear`; `hysteresis(z, enter, exit)`: flat → ±1 when |z| ≥ enter; a held side exits when `z × side ≤ exit`, and may re-enter the other side the same session if |z| ≥ enter; NaN z → flat. **ELB state** `elb_state(model panel)` = policy rate at its floor (`r0 − elb ≤ 1e-9`) **and** the rule's notional below it (`at_elb`): both on the floor, so the model path is flat and has no view. Treatments: `flat` (position 0 in the state, hysteresis reset to flat on exit from the state, close/open costs charged, statistics over non-ELB sessions), `exclude` (the `hold` run's P&L restricted to non-ELB sessions, no boundary costs), `hold` (positions through the state; statistics over all sessions, ELB-session P&L on its own line). A cross sleeve is in the state when either currency is. The state moves with r* (it is model-dependent): say so. Carry filter option (diagnostic). Sizing modes for sleeve-level evaluation: `unit` (±1 book DV01) and `vol_scaled` (`q = s × σ_target / σ_t`, σ from `risk.py`, lagged, floored), with the `risk.no_trade_band` applied to rescaling.
- `strategy/costs.py`: one-way cost per unit |ΔDV01| in bp per leg from config: futures `ticks_round_trip × tick.other/0.01 / 2 + fee_per_side / dv01_contract` (2 ticks: 0.5bp + 0.024bp); others `round_trip_bp / 2`. Charges: same instrument `|q_t − q_p| × one_way`; instrument change (ZQ month, forward window) `(|q_p| + |q_t|) × one_way`; par legs `2|q| × one_way` every `roll_every_months` since entry while held. A uniform-override mode sets every leg's round trip to `c` bp (fee included in `c`: say so) and an assumed-only mode varies only the `observed: false` legs.
- Metrics used here (Sharpe = mean/sd of daily P&L over the evaluation sessions, flat days included, annualised by observed sessions per year; SE(SR) = sqrt((1 + SR²/2)/years)).
- `scripts/build_strategy.py` → `reports/costs.md`, `reports/figures/cost_sensitivity_{light,dark}.png`, `reports/figures/hysteresis_grid_{light,dark}.png`, `reports/results/costs.json`:
  - **Cost sensitivity curve** per sleeve (vol-scaled primary, unit secondary) and for the equal-risk sum of sleeves (labelled: the book proper is week 9), net Sharpe vs uniform round trip, drawn to max(5bp, 1.2c*), with the assumed cost marked (the mix point separately) and the **breakeven cost in closed form** `c* = 2 × gross P&L per year / DV01 traded per year` (maintenance included), "none (gross ≤ 0)" when gross ≤ 0. A second curve varying only the assumed legs (USD ZQ fixed at its tick cost) shows the observed-vs-assumed asymmetry; state it in words.
  - Turnover per sleeve per year: signal turnover (same-instrument changes), maintenance (rolls, par re-strikes), entries a year, average holding period (sessions of constant non-zero side), time in market. Curve sleeves at `roll_every_months` 1 vs 3 as a stated sensitivity.
  - Hysteresis grid (`positions.grid`, `exit < enter`): net and gross Sharpe and turnover per cell (tables + heat map), the chosen (1.0, 0.0) marked; neighbours = up to 8 adjacent cells with exit < enter; report min/median/max net Sharpe over neighbours and over the grid, share of cells > 0, first-half vs second-half at the chosen cell; the generated sentence may say "stable" only if the neighbour range is < 1 SE. The pair was fixed in config before the grid ran: DECISIONS row with the date.
  - Pre- vs post-cost Sharpe for linear and hysteresis rules, under each ELB treatment (say which treatment every number uses).
- Replace the stale `backtest.tags.costs` in both currency blocks with the real cost tags (USD observed tick-based, GBP assumed — the asymmetry).
- DECISIONS rows: costs, the pair, the ELB state and treatments (proposed), sizing modes, and **the post-cost results with an honest reaction**, written plainly from the numbers (proposed; the author's to own).

---------------------------------------------------------------------------------------------------

## 5. Week 9: model variants, portfolio, robustness

### 5.1 Model-side variants (optional config keys; baseline bit-identical)
- **Plumbing first** (else overrides are silently dropped): `model.path.inputs(ccy, root, rule=None)` reads the merged rule block; `market.extractor(ccy, root, calendar, cfg=None)` and `panel.build(..., cfg=None)` take a merged config; `regress.outputs` passes its merged `cfg` through. With no override the objects are built exactly as today (bits unchanged). Tests: an HLW override reads `nyfed/HLW_RSTAR`; an nss override changes the GBP fixture paths; no override leaves them identical.
- **r\***: `rule.rstar: {source: nyfed, series: HLW_RSTAR, label: HLW, real: true, before_first: 2.0, before_first_label: "Taylor's 2%"}`; `real: true` = already a real rate, do not subtract the target. Constants via `{constant: x}`: USD = the full-sample mean of the SEP-based r* (hindsight, like GBP's); GBP = baseline ±1pp (−2.6, −0.6). GBP HLW: n/a (real-time UK vintages end 2020Q2).
- **Coefficients** (`model/estimate.py`, flagged author-adjacent): partial-adjustment form consistent with the imposed inertia. At quarter-end sessions: `R*_impl = (R_q − ρ R_{q−1})/(1 − ρ)` (ρ = `rule.inertia`, per quarter), `y = R*_impl − r* − π`, `X = [π − π*, −(u − u*)]`, no intercept, over quarters strictly before the session's quarter where neither q nor q−1 is in the ELB state; **shrunk toward the imposed values**: `β = (X'X + K)⁻¹ (X'y + K β0)`, `β0` = the config coefficients, `K = n0 × diag(mean(X²))`, `n0 = 8` quarters. Two sentences: "starts at the balanced approach and moves toward the data as off-floor quarters accumulate; the prior counts as two years." Report the raw OLS path next to it (the evidence for R1, "why imposed"). The robustness cell is also reported over the post-first-estimate window only.
- **Z window**: 365D, 548D, 1095D; `min_periods` = round(250 × days/730).
- **Curve fitting** (GBP): `market.curve.method: nss`: NSS by profile least squares, τ1 ∈ geomspace(1/12, 5, 25) years, τ2 in the same grid with τ2 ≥ 1.5τ1, β by batched OLS; count boundary optima, report per-day node RMSE; keep `pin_first_regime`. **Marks stay on the Bank's curve** (only the signal changes). USD n/a.
- **Conditioning**: `rule.conditioning: {converge: {half_life_quarters: 4}}`: `x_k = x* + (x_0 − x*) × 0.5^(q_k/4)` for π (x* = target) and u − u* (x* = 0), `q_k = (E_k − session).days / 91.3125`; notional and floor per meeting **with every dated schedule resolved at `as_of`** (`on(…, as_of)`), r* and spread flat; `reaction.inertial_path` accepts an array target (scalar path unchanged bit for bit). Test: a schedule change between as_of and E_k does not move the path. Model-panel summary columns stay scalar.
- Look-ahead: synthetic test for `estimate.py` (coefficients at sessions ≤ D unchanged when later quarters are poisoned).
- DECISIONS: the OBR u* promise in X11 and the GBP config comment ("a week 9 job") is deferred with a row saying why; update the comment.

### 5.2 Portfolio (`strategy/portfolio.py`)
- **Book calendar**: the book currency's sessions (`book.currency`). A sleeve's P&L on its own sessions is summed into the next book session; positions change only on book sessions (a sleeve on another calendar executes at its next own session).
- **Covariance** of sleeve P&L (vol-scaled-free: unit P&L), strictly before t: EWMA weights `w ∝ λ^age` normalised; `S0 = Σ w x x'`, one-lag non-synchronous term `S1 = Σ w x_t x_{t−1}'`, `S = S0 + S1 + S1'` (London closes before New York); shrink toward its diagonal: `π_ij = Σ w (x_i x_j − S0_ij)²`, `n_eff = 1/Σ w²`, `δ = clip(Σ_{i≠j} π_ij / n_eff / Σ_{i≠j} S_ij², 0, 1)`, `Σ = (1−δ) S + δ diag(S)` (a Ledoit–Wolf-type intensity with the diagonal target, Schäfer–Strimmer's "target D": name it accurately). Report δ over time and the condition number; report MV and ERC with and without shrinkage (the headline inverse-vol does not depend on it). Do not pre-write the conclusion.
- **Sizing signal** `g_s = s_s` (the hysteresis side). Constructions on active sleeves: **inverse_vol** (headline, proposed) `w_s = g_s / σ_s`; **ERC** with signs fixed by `g` (cyclical coordinate descent on `S Σ S`: `v_i = (−c_i + sqrt(c_i² + 4 Σ̃_ii b))/(2 Σ̃_ii)`, tolerance 1e-10; one active sleeve ⇒ ERC = inverse-vol); **mean_variance** `w = Σ⁻¹ μ`, `μ_s = s_s × min(|z_s|, z_cap) × σ_s` (the plan's "z as expected return"; note in the docstring that MV = D⁻¹R⁻¹(μ/σ) and equals inverse-vol-weighted z when correlations are zero).
- **Vol target**: ex-ante `sqrt(w'Σw × periods) = vol_target × capital`, σ floored (`risk.sigma_floor`), gross DV01 cap (`risk.max_gross_dv01_per_capital`, set with a row), `no_trade_band` on each leg's DV01. Daily rebalance otherwise; costs on every leg's change; report turnover split into entries/exits, resizing, rescaling; realized vs target vol by year.
- **Drawdown control**: `dd_t = (max_{u≤t} E_u − E_t)/capital`; scale to 0.5 when dd > trigger (0.10), back to 1 when dd < release (0.05), decided at t, applied with the execution lag. Report return, Sharpe, max DD with and without, the cost in return, the number of trigger episodes (≤ 2 ⇒ the generated sentence says anecdotal). Not used in robustness cells or cost curves.
- **RV-only book** variant (curve + cross sleeves, no outrights) instead of a DV01 projection; and net DV01 by currency of the headline book over time.
- `min_history` sessions of sleeve P&L before the book trades.
- Reports: `reports/portfolio.md`, figures, `reports/results/portfolio.json`: the three constructions side by side (gross/net Sharpe, SE, vol, max DD, turnover, time in market), shrinkage on/off, drawdown on/off, RV-only book. Headline book's cost-sensitivity curve with its breakeven cost.

### 5.3 Robustness grid (`strategy/robustness.py`, `scripts/build_robustness.py`)
One-at-a-time departures from the baseline: r* (USD: SEP\*, HLW, constant; GBP: −1.6\*, −2.6, −0.6), coefficients (imposed\*, estimated), z window (365D, 548D, 730D\*, 1095D), curve fitting (GBP: log-linear\*, NSS), conditioning (hold-flat\*, converge), plus slope definition (orthogonal\*, raw 8−1) and carry filter (off\*, on). **Invariant: P&L marks, instrument selection, carry, roll and costs always come from the baseline instruments; a variant changes only the signal inputs** (test: under NSS the unit P&L series is bit-identical to baseline). Reuse the baseline unit P&L across rows; rerun only model path + signal (+ the GBP market path for NSS), from `data/panel/` (sessions, meetings, macro). Cells: primary = net Sharpe of the headline book without the drawdown overlay, on the **common sample** (sessions where every row has an eligible signal), with SE(SR) in the header; secondary tables: gross Sharpe, each row on its own sample, per-sleeve IC(21) on rate changes. r* rows also: change in mean and latest gap_bp at k=4, corr(z, baseline z), share of sessions whose side changes. Say it is one-at-a-time (interactions untested). Output `reports/robustness.md`, `reports/results/robustness.json`, a generated range line per choice. **Present the chosen spec with the others around it; never pick the best cell.** The prose paragraph goes in the note.

---------------------------------------------------------------------------------------------------

## 6. Week 10: attribution, metrics, tear sheet

- `backtest/metrics.py`: `sharpe` (§4 definition), `sharpe_se`, `hit_rate` (daily and per trade, net of costs), `max_drawdown`, `turnover`, `time_in_market`.
  - `ic(z, x, horizons, lag)`: `fwd_t(h) = Σ_{j=1..h} x_{t+lag+j}` with `x` = the **rate-change component** of unit P&L (total unit P&L as a second row); IC = mean of the product of standardised ranks of `z_t` and `fwd_t`; t = IC / NW-SE with Bartlett lag `h`; cross-check on non-overlapping samples (mean and range over the h start offsets); on non-ELB sessions and on all.
  - `decompose(book)`: P&L by component group — level (outright sleeves), slope (curve), cross-country — **and** the level-factor view: each sleeve's unit P&L beta and R² on its currency's outright unit P&L; the share of book P&L explained by the level factor; corr(level z, slope z) and sign agreement. A generated sentence: if level dominates, say so.
  - `carry_vs_rate`: book and sleeve P&L split into carry+roll vs rate change, and a benchmark with position = sign(CR_h) under the same hysteresis and costs; correlation of the strategy with it ("a carry trade in costume?").
  - `regimes(policy rate path, elb_state)`: one real-time state machine per currency, used by weeks 10 and 12: ELB (elb_state) overrides; `early_hiking` (< 12 months since the first hike of the cycle, no cut since); `late_hiking` (≥ 12 months, no cut, a hike in the last 6 months); `hold_after_hikes` (no hike for 6+ months, no cut); `cutting` (a cut in the last 6 months); `hold_after_cuts` (otherwise). Hiking = early + late for the table. Cross sleeve and book: keyed by USD's state, and the pair (same/different). Active sessions and trade counts next to every regime Sharpe. (The USD panel shows a hike on 2026-09-17: a new cycle with no observations.)
  - ELB comparison table (flat\* / exclude / hold): P&L in ELB sessions, outside, Sharpe outside, overall, boundary trades. Limitations row: the post-lift-off z is standardised against ELB-era windows for two years.
  - `ex_2022(result)` and `ex_2022_23(result)`: named functions (windows from `evaluation.exclude`), statistics on the P&L with the window removed (positions not re-run), for sleeves and book.
- `report/charts.py` completed (equity by component, cost sensitivity, IC bars, regime bars) and `report/tearsheet.py` + **`scripts/build_tearsheet.py`** (one command from `data/panel/` and the cache; runs the strategy chain in memory) → `reports/tearsheet_<date>.pdf` (+ `--preview`): framework in three sentences, headline chart (book equity net of costs, by component), current signal table, performance summary (gross/net Sharpe with SE, vol, max DD, hit rate, turnover, ex-2022 and ex-2022-23, per sleeve and book), the headline book's cost sensitivity curve with its breakeven. One page, generated. `reports/results/metrics.json`.

## 7. Week 11: the note (draft)

- `notes/note.md` (the author's prose to rewrite) with `{placeholders}` for numbers and `{table:...}` for generated tables; `scripts/build_note.py` fills them from `reports/results/*.json` and the generated tables into `reports/note.md`. A bare number with a unit in the prose is flagged as a **warning** by build_note.py (not a pytest failure).
- Plan's 10 sections, six to ten pages, buy-side register (claim, evidence, trade, the conditions to abandon the view). First line: "Draft 1: agent-drafted from the plan; every word is the author's to rewrite before it is sent."
- `notes/gaps.md`: every gap the draft exposes, each *fixed* (what changed) or *moved to limitations* — and every "moved" gap becomes a `tags` entry so the generated limitations include it.

## 8. Week 12: credit bridge

`src/policypath/credit.py` (+ `report/credit.py`, `scripts/build_credit.py`, `tests/test_credit.py`), for every enabled currency with a `credit` block (USD). Primary spread **Baa − Aaa** (maturity-matched); DBAA − DGS30 and BAA10Y alongside; ICE IG/HY OAS 2023-09 on as a short cross-check. Spread changes proxy excess returns: say so. Report the AR(1) of weekly Δspread and its cross-correlation with lagged ΔDGS10 before test 2 (seasoned-bond staleness); if staleness shows, add lagged ΔDGS10 as a control. Weekly = last observation on or before each Wednesday (Fed calendar); numpy OLS; Newey–West Bartlett.
1. **Contemporaneous**: weekly Δspread on Δmarket_k4, Δmodel_k4 (sum = Δgap), Δslope (orthogonal), Δdifferential, all bp, with and without Δ(DGS30 − DGS10); NW lag 4. R² of Δmarket alone = "how much of credit's variation is a repricing of the policy path".
2. **Predictive**: `s_{t+1+h} − s_{t+1}` on gap_t (bp and z), h = 21 and 63 sessions, NW lag h, non-overlapping (every h-th session) cross-check; with and without 2020-02-15…2020-04-30; state whether ELB sessions are included. **Pre-register the expected sign in DECISIONS before running**: gap < 0 (tightening underpriced) predicts widening, a negative slope.
3. **Conditional**: slopes on the level component by the §6 regime state (early vs late hiking, etc.): a per-episode table (2016 early, 2017–18 late, 2022 early, 2023 late) with NW SEs next to the pooled interaction; wording "the sign differs / does not differ in each of two cycles"; a DECISIONS row that power is too low for a nominal-size test.
Outputs `reports/credit_USD.md`, one chart `reports/figures/credit_USD_{light,dark}.png`, `reports/results/credit.json`, and `notes/credit_section.md` (~2 pages, included by the note). Quirks go into the credit block's tags. A clean null is a result.

## 9. Week 13: finish

- `report/limitations.py`: the limitations section **generated from every tags block** (both currencies, strategy.yml), grouped by kind (lag, quality, modelling), deduplicated across currencies; the note (`{limitations}`) and README use it; the brief footer keeps its block-scoped subset. Check every DECISIONS L-row and every gaps.md "moved" item has a tag.
- README rewrite: the claim, the headline chart, reproduction (fresh clone → `uv sync` → key in `.env` → `update_data.py` per currency → build scripts in order → tear sheet), data licensing stated exactly (CME via Databento: bring your own licence; small test fixtures of CME settles *are* committed — say so, and why; ICE BofA via FRED truncated to three years; Moody's via FRED; BoE, ONS, ALFRED, NY Fed public).
- Hygiene: clean install in a fresh environment from the **working tree** (copy tracked + new files, not `data/`, to a scratch dir; `uv sync`; `uv run pytest`) — the real fresh clone happens after the author commits; no dead code, no commented-out blocks, no `_v2` names; `reports/` has a current dated tear sheet and brief.
- Git history check: list every raw-data path ever committed (`git log --all --name-only`), in HEAD and in history. **Report to the user with options** (keep under a licence review; or replace with synthetic strips + refreeze + a history rewrite, which needs the user's explicit authorisation). Change nothing. Also note the tracked `.DS_Store`.
- `notes/verbal.md`: the 90-second version (claim, method, result, biggest weakness), written to be said aloud. Rehearsing it is the author's.

## 10. DECISIONS.md sections and IDs

`## Expression and carry (week 7)` C1…; `## Costs (week 8)` K1…; `## Model variants and robustness (week 9)` V1…; `## Portfolio (week 9)` P1…; `## Attribution and evaluation (week 10)` A1…; `## The note (week 11)` N1…; `## Credit bridge (week 12)` D1…; `## Finish (week 13)` F1…; Phase 0 adds R10 (EFFR 2016 methodology) and L12… rows for every new series in the lags table. Each new section starts with: "Drafted 2026-09-28 by the coding agent from the week 7-13 plan; for the author to confirm or rewrite."

## 11. Departures from the plan, and why (each gets a DECISIONS row)

1. **The plan's week-7 carry example has its sign backwards.** Paying on an inverted front end *earns* carry and roll. "Right and bleeding" is **receiving on an inverted curve** (the rule wants even more cuts) or **paying on an upward-sloping one**. And bleeding is a mark-to-market cost in the standing-still scenario, not an expected loss if the model is right.
2. **The plan's claim that the identity proves the signs is too strong**: carry + roll + rate = total telescopes. The defence is the identity plus an independent revaluation, a convergence test and the GBP end-to-end reproduction (§1.3).
3. **"Gap" in the breakeven means the de-meaned gap** (the edge), not the raw bp gap; and "the gap covers the carry" becomes "the expected quarter pays for the bleed" (§1.6), which does not double count.
4. **Credit data.** ICE BofA OAS on FRED/ALFRED start 2023-09-29. Moody's Baa − Aaa (daily since 1986) is primary; ICE a 3-year cross-check; no free long HY series.
5. **GBP 2s10s from gilts**: the Bank's OIS curve reaches 25y only from 2016. USD uses Treasury CMT: both curve sleeves are government 2s10s. The cross-country sleeve is the plan's own example, **2y vs 2y** (government par yields), which keeps it distinct from the outright instruments (a cross built from the outright legs makes the sleeve covariance singular).
6. **The slope component is orthogonalised against the level** (it is 85–94% the level otherwise); the raw 8−1 is a robustness row.
7. **The ELB decision** is taken as a model-dependent "no view" state (policy rate and rule both on the floor), implemented from week 8 so every week's numbers use one stated treatment, with all three options shown in week 10.
8. **HLW** from the NY Fed's real-time vintages (2015Q4–2020Q2, 2022Q4–2026Q2); UK real-time vintages end 2020Q2, so GBP HLW is n/a.
9. Acceptance items only the author can do (read aloud, rehearse, the real fresh clone after a commit) are listed as open author actions.

## 12. Order, ownership, cut order

Order: **Phase 0 → {Week 7 ∥ Week 12 ∥ Week 9 model variants (§5.1)} → Week 8 → Week 9 portfolio + robustness (§5.2–5.3) → Week 10 → Week 11 → Week 13.**

File ownership in the parallel phase:
- Week 7: `signal/components.py`, `strategy/{__init__,instruments,expression,carry}.py`, `report/expression.py`, `scripts/build_expression.py`, `report/brief.py`, `scripts/build_brief.py`, `config/brief.yml`, tests `test_components.py`, `test_instruments.py`, `test_expression.py`, `test_carry.py`, and a shared-helper extraction in `curves/forward.py` (bit-identical). DECISIONS section C.
- Week 12: `credit.py`, `report/credit.py`, `scripts/build_credit.py`, `tests/test_credit.py`, `notes/credit_section.md`, the USD `credit` block of `config/currencies.yml` (that block only). DECISIONS section D.
- Week 9 model variants: `model/{path,reaction,rstar,estimate}.py`, `market.py`, `panel.py`, `regress.py` (plumbing), `curves/forward.py` (NSS; coordinate: add functions, do not touch the helper week 7 extracts), the `rule`/`market.curve` keys of `config/currencies.yml` and `config.py` for them, tests `test_variants.py` (or `test_estimate.py`, `test_nss.py`, `test_conditioning.py`). DECISIONS section V.
- `notes/DECISIONS.md` and `notes/author_review.md`: each agent adds only to its own section, with `Edit`.

Amendments made at launch of the parallel phase: NSS lives in a new module `curves/nss.py` (week 9 does not edit `curves/forward.py`); the ELB state (`elb_state`, §4's definition) and the regime state machine (§6) live in a new shared module `src/policypath/regimes.py`, written by the week 12 chain; weeks 8 and 10 import them from there rather than defining them in `positions.py`/`metrics.py`.

Cut order if time runs short (plan): credit bridge first, then the ERC/MV variants (keep the headline construction). Never the note, the brief, the tear sheet or the generated limitations.
