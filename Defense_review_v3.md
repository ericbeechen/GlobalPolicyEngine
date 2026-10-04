# Defense.docx: full review of the 2026-10-04 draft

This review covers the 12:45 version: 14 pages, about 2,850 words, 13 figures, 2 tables. Numbers are checked against `reports/numbers.md` and the reports it links to.

It is ordered from the whole paper down to single lines:

1. Where the paper stands
2. Problems that span sections
3. Section-by-section scorecard
4. Line edits, with suggested rewrites
5. Length and layout
6. Finishing order

---

## 1. Where the paper stands

**The argument is now complete.** Every claim in the spine has a section that measures it, and every number I checked matches numbers.md. The last round's factual errors are fixed: the ρ units, the reversed "market leads the Fed" bullet, +36, u\*, the level gross and net, and the coefficient mix-up.

What's left falls into three kinds of work:

1. **Framing.** A few things a sharp reader would challenge before reaching the results (§2): the title's claim about GBP, an undefined vocabulary, and no references.
2. **Order.** Two sections state a conclusion before the evidence for it (Costs, and the P&L paragraph in The level).
3. **Polish.** Voice, filler and leftover pre-revision sentences. These are the main thing between this draft and a finished one.

By section: Summary, Construction (USD), Signal, Performance and What would change my view are close to done. Why this question, Carry filter and Robustness still need a real pass.

---

## 2. Problems that span sections

### 2.1 "The central bank's own rule" holds only for USD
The title, the Summary's first sentence and Finding 1 all say *the central bank's own rule*. The BoE publishes no balanced-approach rule. For GBP the paper applies **the Fed's rule with UK inputs** (`config/currencies.yml`: "The same rule as USD, imposed, with the Bank's own numbers where they differ"). A reviewer will catch this on page 1.

The fix is a sentence, not a restructure. In *Why this question* or *Rule, r\**, add: "The BoE publishes no such rule; for GBP I apply the same rule to UK inputs, so the GBP comparison is to the Fed's rule, not the Bank's." Then make the title and Finding 1 consistent. Two options:
- **(a)** "Does the market price the Fed's own rule?", treating GBP as the comparison case;
- **(b)** keep the title but say "a published policy rule" in Finding 1.

### 2.2 The GBP rule's inputs are never described
*Real-time inputs* describes only USD. GBP uses:
- **headline CPI** (12-month, the MPC's target, never revised), not core PCE;
- **LFS unemployment** as first printed;
- u\* constant 4.5%;
- r\* constant −1.6%.

Using headline in one currency and core in the other is a choice a reader will want named. Add two sentences.

### 2.3 Acronyms are never defined
ZQ, SR1, EFFR, SOFR, OIS, MPR, SEP, HLW, ELB, DV01, IC, LFS and CBO all appear undefined, or are defined only after first use. The Definitions table is the natural home. Add one-line rows, or define each in the text on first use:
- **ZQ:** CME 30-day fed funds futures
- **SR1:** CME one-month SOFR futures
- **EFFR:** effective fed funds rate
- **SOFR:** Secured Overnight Financing Rate
- **OIS:** overnight index swap
- **MPR:** Monetary Policy Report
- **SEP:** Summary of Economic Projections
- **HLW:** Holston–Laubach–Williams r\*
- **ELB:** effective lower bound

Also: **"SEP (Statement of Economic Projections)" is the wrong name.** It is the *Summary* of Economic Projections.

### 2.4 Figures are never referred to in the text
The only reference to a figure is "The figure shows" in Breakeven. Each figure should be called by number at the point the text relies on it, e.g. "(Figure 5)". Use Insert → Cross-reference → Figure so the numbers follow if you move things.

The outline's rule for sections 4–6 was "place the chart first and write to what it shows". Right now the charts sit beside the text rather than being read by it.

### 2.5 Voice
The text uses four voices:
- **"the paper"** (about 10 times): "the paper uses", "the paper tested", "the paper seeks", "the paper's view";
- **commands**: "compute…", "solve…", "leverage…", "estimate the variables", "Hold today's inputs…";
- **"you"**: "you can take 19.8 turns…";
- **"I"**: "That's why I don't claim it", from the plan, though that line didn't make it in.

Pick **"I"** (single-author defense) and convert the rest. Searching the doc for *the paper*, *compute*, *solve*, *leverage*, *estimate*, *Hold* and *you* will find nearly all of them.

### 2.6 No references
Name these in a short References list after Limitations:
- Taylor (1993);
- the Fed's Monetary Policy Report (the balanced-approach and inertial rules);
- Holston, Laubach and Williams (HLW);
- CBO's natural rate of unemployment, via ALFRED (St. Louis Fed);
- the BoE yield-curve data and MPR conditioning paths;
- CME contract specifications for ZQ and SR1.

### 2.7 There is no ending
The paper goes Robustness → What would change my view → Limitations → Appendix. Nothing restates the result after the challenges. Add two or three sentences, either as the first lines of *What would change my view* or as a short "Conclusion" before it. They should say that the null survived every challenge, and that GBP outright is the one open question, which is why the list below is mostly about it.

### 2.8 Typography
- **Minus signs:** the tables use the true minus (−) and the text uses hyphens (-0.12). Find and replace "space-hyphen-digit" with "space-minus-digit", or leave hyphens and accept the mismatch.
- **Arrows:** "->" appears three times. Use → (in Word, type 2192 then Alt+X).
- **Sharpe:** "Sharpe Ratio", "Sharpe ratio" and "Sharpe" all appear. Pick one; "Sharpe ratio" or "net Sharpe" is the convention.
- **Dates:** "1/13/2014 to 10/01/2026" sits next to "April 2014 to September 2021". Use "13 January 2014 to 1 October 2026".
- **Units:** bp is now consistent. Keep it that way.

---

## 3. Section-by-section scorecard

| Section | Job | State | What remains |
|---|---|---|---|
| Summary | Four findings with numbers; it's a null | **Nearly done** | Finding 1 has no number; Finding 3 doesn't say *which* entries; Finding 4 still says "Roll maintenance"; headline book line missing "gross" and the drawdown |
| Why this question | The benchmark can't be gamed; anyone can audit it | **Weak** | The core argument (no researcher degrees of freedom, auditable) isn't stated; "creates a strong foundation" is filler; the GBP point (§2.1) belongs here |
| Definitions | Lookup | Good | "Summary", not "Statement"; GBP r\* "except near the floor"; acronyms (§2.3) |
| Construction: USD | The market side is measured | **Done** but for polish | Commands in the SR1 paragraph; one run-on sentence; a doubled "to address this" |
| Construction: GBP | Same, for GBP | Good | Commands; logic order of the curve sentence |
| Rule, r\* | The rule side is real-time | **Half revised** | The old "resource differences / divergence" sentence now contradicts the new "symmetric" one; no path equation (so the ELB floor is never shown); GBP inputs missing (§2.2) |
| The level | Report the level as a measurement | Good | GBP logic sentence; a missing "as"; the P&L paragraph belongs in Performance (§4) |
| Signal and the null | The core null; GBP is the exception | **Good** | The pun; "solid… solid"; one garbled sentence; no closing line restating the null |
| Performance | Book and sleeves; GBP's caveats | **Good** | Missing subject in paragraph 1; "fierce"; GBP SE; no "I don't claim it" |
| Reaction | Nothing re-tuned | Done | Optional: name the pair (1, 0) |
| Costs and the roll | Costs follow turnover; half are structural; no rescue | **Out of order** | The conclusion comes before the evidence; re-strikes never explained; Figure 10 not referenced |
| Breakeven | The identity; who clears | Good | A vague sentence; paragraph order should follow the figure |
| Carry filter | Statistically real, economically irrelevant | **Still weak** | The screening sentence is still garbled and reversed; no SE; no "pre-specified" |
| Robustness | The null survives every single challenge | **Still weak** | The first two sentences overclaim and repeat the third; the GBP "choice" wording; the paired-SE point missing |
| What would change my view | Concede what would overturn it | **Done** | Split one combined bullet |
| Limitations | Honest list | **Regressed** | "GBP r\* is a hindsight constant" was *replaced* by the u\* bullet rather than joined by it; restore it |
| Appendix | Reference material | Started | See the list in the writing plan, §11 |

---

## 4. Line edits, with suggested rewrites

### Summary
- **Paragraph 2, last sentence:** "All of these are after a year of history and outside the ELB" merges two different conditions. Suggested: "z needs a year of history first, and the IC is measured outside the ELB."
- **Finding 1:** add the number. "…below the central bank's own rule: in USD, in every one of 13 years, by 25 to 86bp at the fourth meeting."
- **Finding 2 (optional, but it's your most interesting result):** "…in USD it points the wrong way (IC(21) −0.12), because the Fed moves toward the market, not the reverse."
- **Finding 3:** "A carry filter that skips entries **whose expected quarter, carry and roll included, is below zero** would improve…"
- **Finding 4:** "**Maintenance (rolls and re-strikes)** is 46% of the cost; rolls alone, forced by trading the 4th meeting ahead, are 33%."
- **Headline, Book:** "Gross Sharpe −0.14, net −0.60 (SE 0.32); −3.19% of capital a year; worst drawdown 57.5%."

### Why this question
This is the weakest remaining section. Suggested rebuild, in three beats:
1. **The comparison.** The market's implied path against where one of the rules the Fed publishes in its Monetary Policy Report would take policy, using only what was known that day.
2. **Why this benchmark.** "The reference is the central bank's own published rule, not one I built, so there are no free parameters to tune and anyone can audit the comparison. The result can't be blamed on the benchmark." Then the GBP caveat from §2.1.
3. **Why the answer matters.** Keep your current second paragraph. Add a pointer: "(The signal and the null)".

Cut "This paper's model is derived from this rule and creates a strong foundation." It says nothing that beat 2 doesn't.

### Definitions
- **USD r\*:** "Statement" → "Summary".
- **GBP r\*:** "This is a hindsight number; however, it moves the bp gap and mostly leaves z unchanged" → "A hindsight number. It moves the bp gap and mostly leaves z unchanged, except near the floor." Robustness shows z correlates only 0.88 with the chosen z at −2.6%, because the floor binds.

### Construction: USD
- **The equation sentence:** "each contract month is one linear equation in the regime rates equal to the share of the month's days" → "each contract month is one linear equation in the regime rates, **each weighted by its share** of the month's days."
- **Doubled phrase:** "…which is necessary as a regime spans several contract months and each month mixes several regimes. **To address this,** they must be solved together…" → delete "To address this,". The sentence before already gives the reason.
- **SR1 paragraph, rewritten in one voice and with the run-on split:**
  > To check the result, I compare it with SR1, the one-month SOFR future. For each session I average the ZQ-implied EFFR path over each SR1 contract month and solve for the constant spread that reprices the SR1 settle: the SOFR − EFFR basis the two markets imply together. I then compare it with the basis later realized. Over 2,116 sessions it misses by +0.52bp on average, 1.86bp in absolute terms, and is within 3bp on 80% of them. Two independent futures markets agree to within about 2bp, so the ZQ path is not an artefact of the solver.
- **Figure 2 caption:** the closing period is missing.

### Construction: GBP
- **Curve paragraph:** "For GBP, leverage a fitted OIS spot curve published from the BoE… Once fitted, expected rates between two meetings is the forward over that window. The log discount factor is interpolated from these." The steps are in the wrong order and the verbs don't agree. Suggested:
  > The Bank of England publishes a fitted OIS spot curve. I interpolate its log discount factors, and the expected rate between two meetings is the forward over that window.
- **MPR paragraph:** "(or 87 quarters)" → "covering 87 quarters". "The GBP path built is" → "My GBP path is".

### Rule, r\*
- **Delete** "The key difference is an increase in the coefficient associated with resource differences; this creates a more explicit emphasis on labor market divergence in comparison to the traditional Taylor rule." Your new next sentence ("2 on the unemployment gap… symmetric") says it correctly, and the two now contradict each other.
- **Tighten** "The balanced approach rule takes a very similar approach to policy setting that the Taylor rule would suggest" → "It is a variant of the Taylor rule."
- **Put the equations on their own lines**, centred: the notional rate, then the path.
  - R\* = r\* + π + 0.5(π − 2) + 2(u\* − u)
  - R_k = 0.922·R_(k−1) + 0.078·max(ELB, R\*), with R_0 the rate in force

  The second line is where the ELB enters the paper; without it, "ELB state" in Definitions has no anchor.
- **Last sentence:** "Hold today's inputs as flat across the path and ask: where does the rule take policy?" → "Today's inputs are held flat across the path, so the rule answers: if nothing changes, where does policy go?"
- **Add the GBP sentence** (§2.1 and §2.2).

### Real-time inputs
- "With the balanced approach rule in mind, estimate the variables." The inputs aren't estimated; they're read as published. Suggested: "Each input is what was published on the day."
- "RMSE of 0.075% pp" → "0.075pp".
- "appends the deliberately corrupted data" → "appends deliberately corrupted data after D".
- Add the GBP inputs (§2.2).

### r\* choices
- "A key benefit of using SEP is how it is at a moment in time. There is no revision, and so it's real time with no vintage." Suggested: "Each SEP is a dated document that is never revised, so it is real time without any vintage machinery."
- Add the results: "…HLW in real time for USD (book net −0.52) and constants ±1pp for GBP (−0.58 and −0.63)."

### The level
- **GBP logic.** "This isn't persistent but depends on regime. That can be seen as 8 of the 11 years remain below the rule, and above from 2024." Being below in 8 of 11 years doesn't show regime dependence; the regime split does. Suggested: "It isn't persistent; it depends on the regime: −81bp while hiking, +28bp while cutting. It sat below the rule in 8 of 11 years, then above from 2024 (+36, +19, +102bp)."
- **A missing "as":** "the post 2024 flip is consistent with r\* being wrong as with…" → "is **as** consistent with r\* being wrong as with the market repricing".
- **Move the P&L paragraph and Figure 7 to Performance**, after the GBP outright paragraph. *The level* is a measurement section, and this paragraph uses book P&L before the reader has seen the book's headline. In Performance it becomes "where the loss comes from". Its last line, "costs alone turn the level negative", then hands off directly to *Costs*.
- **Same paragraph:** "the rest lost −0.77%" → "the rest lost 0.77%" (a double negative). Add the punchline: "The risk is the level's; the loss is the rest's."
- 63% is still DECISIONS A4, a *proposed* reading. You're quoting it, so you own it. Fine, as long as that's deliberate.

### The signal and the null
- **Cut the pun:** "The GBP is the exception to this, not the rule."
- **Repetition:** "The results are solid in its case, with GBP outright seeing a solid +0.38" → "GBP outright is the exception: IC(21) +0.38 over 2,273 sessions, t +4.7."
- **Garbled sentence:** "carry little information as is present from their t stats trailing small" → "The three curve and cross sleeves are small (|t| ≤ 1.6)." Be careful with USD 2s10s: its non-overlapping range (+0.02 to +0.17) doesn't include zero, so "small" is right and "zero" isn't.
- **Paragraph 2:** "Additionally, it is important to mention that the t is Newey-West…" → "The t is Newey-West to lag h, but z is persistent, so the products stay autocorrelated past h and the t overstates the precision." "They" was ambiguous.
- **Paragraph 3:** "(t+4.7)" → "(t +4.7)". Add the source: "(DECISIONS V18)". It is a proposed reading, so your text treating it as an interpretation is right.
- **Add a closing line:** "So the gap carries no usable information at the book level; GBP outright is the one exception, and the next section shows why it can't be claimed."

### Performance
- **Paragraph 1:** "Gross −0.14, net −0.60…" → "The book's Sharpe ratio is −0.14 gross and −0.60 net (SE 0.32), −3.19% of capital a year." Cut "fierce": "The worst drawdown, 57.5% of capital, is a grind rather than a break…"
- **Paragraph 2:** add the SE to the +1.76: "+1.76 (SE 0.87) while hiking, on 4 trades." That makes it plain the window is about two standard errors on four trades. Add the numbers to the caveats: "a hindsight r\* (±1pp moves it from +0.67 to +0.47)". End with "That's why I don't claim it."
- **Then the moved P&L paragraph and Figure 7** (see The level).

### Reaction
Optional: "The hysteresis pair (1, 0) was set in config at Phase 0, before the grid ran."

### Costs and the roll
**Reorder.** Paragraph 1 currently opens with "Gross is already −0.72%… even charging rolls at half price…" before the reader knows what the costs are or what a roll is. Suggested order:
1. **The reconciliation.** "Turnover explains the cost exactly: 19.8 turns a year × 284k per bp of mean gross DV01 × 0.44bp one way = 2.47% of capital a year, the cost charged." Drop "you can take", and say "of capital".
2. **Maintenance.** Your paragraph, plus re-strikes: "…the par legs are also re-struck every quarter (13%), which brings maintenance to 46%." Delete "consistent", which nothing supports.
3. **No rescue, the conclusion, pointing to Figure 10.** "Gross is already −0.72% a year, so no cost treatment makes the book pay (Figure 10). Charging rolls at half price saves 0.4% a year (−3.19% → −2.78%) without changing the sign." Also fix "half prices" → "half price" and "insufficient to change to the outcome" → "…to change the outcome".

Robustness offers an extra supporting fact: before costs the book earns in only 4 of 14 specifications, at most +0.13 gross Sharpe. That belongs either here or in Robustness.

### Breakeven
- **Order.** The figure comes first, so lead with the paragraph that reads it: carry and roll ran against 53% of signals, *and those signals didn't earn less*. That second half is what the figure's title says, and it's the surprising part. Then the identity, then the ρ paragraph.
- "the bets that the rules take" → "the rate calls".
- "This is a significant shortfall and is a theme that crosses strategies, although to varying effects." → "3.3 times short."
- "Only GBP − USD 2y collected carry" → "GBP − USD 2y collected carry but was wrong on the rate."

### Carry filter
The section still doesn't do its job. "The filter screens bet to see if the expectation of holding the position is too significant for the expected return" is garbled, and the logic runs backwards. Suggested replacement for the whole section:
> The carry filter was specified in advance as a diagnostic (DECISIONS A15). At each entry it asks whether the expected quarter (the share of the gap that has historically closed, plus carry and roll on the rest) is below zero, and skips the entry if so. It skips 16 of 104 entries (15%). That lifts the book's net Sharpe from −0.61 to −0.51 on the common sample, +0.10 with a paired SE of 0.05. So the effect is statistically real, about two paired SEs, and economically irrelevant: the book still loses, and the filter stays off in the headline.

### Robustness
- **Delete the first two sentences.** "Varying the strategy or choices does not change the outcomes of the study" overclaims. "The Sharpe Ratio remains well maintained at −0.70 to −0.29" repeats the range the third sentence gives.
- **Add the paired-SE point**, which is what makes the range meaningful: "None moves the book by one SE (0.32). Four move it by more than one paired SE, all upward, which is about what chance gives across 13 rows (about 4 expected). The chosen specification was fixed before the grid ran."
- "Even focusing on the best performing choice, GBP outright's sensitivity…" mixes up a choice and a sleeve. Suggested: "The aggregate hides the sleeve: GBP outright, the one sleeve that earns in every row, falls from +0.67 to +0.13 under estimated coefficients."

### What would change my view
- Split "A more harmonized r\* across currencies or calendar-anchored signal which would remove the forced roll" into two bullets. They're unrelated.
- "There are several details that would change the paper's view on this matter. In no order:" → "Any of these would change my view:"

### Limitations
- **Restore** "GBP r\* is a hindsight constant (−1.6%, the 2009–26 average real Bank Rate)."
- **Add:**
  - UK labour data: the LFS was suspended October 2023 to January 2024 and is "official statistics in development" since. The UK input is noisier than the US one.
  - The first z's after lift-off are standardised against a window that is mostly ELB sessions (DECISIONS A8).
- "Overlapping windows inflate t-stats" → add "(the non-overlapping check in Signal addresses this)".

---

## 5. Length and layout

**Length.** You're at 14 pages against the outline's 8, with about 2,850 words and 13 figures. The figures take more than half the page area. Finishing the edits above adds perhaps 300 words. To get near the target, move figures to the appendix. Their numbers are already in the text, so the argument doesn't depend on seeing them:

| Move | Saves | Why it can go |
|---|---|---|
| Figure 2 (SR1 basis) | ~⅓ page | The text quotes all three numbers |
| Figure 3 (GBP path) | ~½ page | Figure 6 shows the GBP market path again |
| Figure 4 (nowcast) | ~½ page | Supporting, not argued from |
| Figure 13 (already in the appendix) | — | — |

That gets you to about 12½ pages. The remaining gap is mostly whitespace from large figures. Shrinking Figures 1, 5, 6 and 12 from 5.6 to 5.0 inches wide would recover another page. If 8 pages is a hard limit rather than a target, the next things to move are Figure 7 and Figure 9's left panel (equity by construction).

**Layout.** Pages 3, 6 and 7 end with large blank areas because a figure didn't fit below the text. These move as the text changes, so fix them last, after the content is final. The cheapest fixes are figure width and moving a figure one paragraph earlier or later.

---

## 6. Finishing order

1. **Framing (§2.1–2.3):** the GBP rule sentence, the GBP inputs, acronyms, and "Summary of Economic Projections". *Thirty minutes.*
2. **The three weak sections:** Why this question, Carry filter, Robustness. Rewrites are suggested above. *One hour.*
3. **Reorder** Costs, and move the P&L paragraph from The level to Performance. *Fifteen minutes.*
4. **Restore** the r\* limitation and add the two new ones; split the combined view bullet.
5. **Voice pass (§2.5)** and typography (§2.8).
6. **Figure references (§2.4)** and References (§2.6).
7. **Closing lines (§2.7).**
8. **Appendix contents** (writing plan, §11).
9. **Length and layout (§5)**, last.
