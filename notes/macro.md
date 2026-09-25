# Macro nowcast: decisions

Numbers are from `reports/nowcast_USD.md` and `reports/vintages_USD.md` (cache as of 2026-09-25).
Tracked, unlike `notes/decisions.md`.

## `as_of` = end of day D

`VintagePanel.as_of(D)` returns every value with `realtime_start <= D <= realtime_end`: what a pull
at the end of D would have returned. It never returns `realtime_end`, which says when a value would
next be revised. Safe against ZQ settle timing: the day's releases (08:30 or 10:00 ET) are out before
ZQ settles that afternoon, so D's settle can already react to D's data; nothing later is visible.

## Activity gap = unemployment gap, u* = CBO NROU vintaged

- `u_gap = UNRATE - NROU`, both from the D vintage, NROU for the quarter of the UNRATE month.
  Positive = slack (the MPR's `2(u_LR - u)` has the opposite sign).
- Not an output gap: potential GDP is quarterly, heavily revised and rewritten in hindsight
  (Orphanides 2001; Orphanides & van Norden 2002: most real-time unreliability is end-of-sample).
- Not Hamilton or a one-sided HP on UNRATE: in real time the 2020 spike sits 24-35 months back in
  2022, and Hamilton reads -2.6pp in April 2022 against -0.85 with CBO, about -0.8pp too tight in
  2025-26. At a coefficient of 2 that is ~160bp of spurious tightening.
- CBO moves little in real time: 8 vintages since 2021, 3 of which moved the gap's quarter on the day
  (2021-02-01 +0.11, 2025-01-17 -0.09, 2026-02-27 +0.09). NROU's first ALFRED vintage is 2011-02-02,
  so the as-of gap exists from 2011 only.
- Sensitivity, reported not used: `u - SEP longer-run median` (UNRATEMDLR) runs 0.1-0.45pp above the
  CBO gap since 2022, i.e. the SEP's u* is lower; 20-90bp in a balanced-approach rule.
- A missing u* raises; it is never carried forward from an earlier quarter.

## Inflation for the rule: 12-month core PCE, bridged from core CPI

- 12-month, not 3-month annualized: the MPR rules use four-quarter core PCE, and a bridged month's
  error enters a 12-month rate once but a 3-month annualized rate about four times.
- Bridge: PCE m/m = a + b x CPI m/m, fitted on the trailing 60 months up to PCE's last month, from
  the same vintage. Real-time RMSE against PCE's first print, 67 months 2021-01 .. 2026-07:
  0.093pp (bridge), 0.132 (CPI 1:1), 0.157 (last PCE carried). Windows 36 / 120: 0.097 / 0.095;
  reported once, not tuned. The slope drifts 0.57-0.75 over the backtest.
- The October 2025 CPI hole is log-interpolated inside the proxy only: the Sep -> Nov change is split
  evenly. Errors on the two bridged months were +0.05 / -0.01pp, smaller than feared.
- Months bridged, days by count: 0: 610, 1: 801, 2: 22, 3: 6.
- Later, out of scope: map CPI components (and PPI airfares, medical, portfolio management) into PCE.

## Standardization: only the activity composite, robust and expanding within the vintage

- Quarter-to-date average over the whole previous quarter, log x 400, per series; the quarter is
  payrolls' latest, falling back one quarter if fewer than `min_series: 3` series reach it.
- z = (growth - median) / (1.4826 MAD), moments over complete quarters from `moments_start:
  1993-01-01` up to but excluding the target quarter, all from the D vintage: inside the per-date
  function, never computed once outside the loop. Median/MAD so 2020Q2 needs no hand-picked exclusion.
- Composite = equal-weight mean of available z's, fixed ex ante. A diagnostic, not a rule input:
  the only z-score in the nowcast.
- GDPNow same-day correlation: +0.46 over 2021-26, -0.25 since 2022, -0.33 since 2022 ex GDPNow's
  2025Q1 (gold imports). Almost all of the full-period number is the 2021 reopening; since 2022
  GDPNow moves on net exports and inventories none of the four series sees. Not reweighted to fit it.

## Series without full vintages, and what was done

1. Real retail sales: FRED's RRSFS is not used; RSAFS / CPIAUCSL is rebuilt as-of.
2. Late vintage starts: every as_of since 2021 is covered; GDPNOW (2016-05-17) binds for the whole
   set, NROU (2011-02-02) for the nowcast inputs.
3. `"."` artifacts in superseded vintages (NROU 2023-07-26, PCEPILFE 2003-2009, CPIAUCSL 1991-94,
   AHE 2015-12-04) are kept as NaN. None touches a value the nowcast reads: NROU's blanks are
   2026+ quarters in a vintage current in 2023-24, and the rest are pre-2016.
4. October 2025 holes (UNRATE, CPI): published as missing, kept NaN. The gap uses November; the
   bridge and the retail deflator interpolate in logs inside the series only.
5. NROU is published ahead of the dates it describes: listed in `projections`, the only exemption
   from the published >= date guard.

## Deliberately not done

No factor model, no Kalman filter, no second currency, no reaction function.
