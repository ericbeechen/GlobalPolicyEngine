"""r*: what the committee itself says the neutral real rate is, as it stood on a date.

The FOMC's median longer-run federal funds rate from each Summary of Economic
Projections, less the 2% target. Each SEP is a dated document published with
the statement and never revised, so it is real-time by construction: step-fill
from one release to the next. The model is of the committee's reaction
function, so what it believes neutral to be is the relevant number, not an
estimate of true neutral (HLW is a week 9 robustness check).

Before the first dot (2012-01-25) there is no committee number, and Taylor
(1993)'s 2% stands in: fixed in 1993, so it carries no look-ahead. FRED rounds
the median to one decimal, as the Fed's own tables have since 2015-09 (2.875 is
2.9), so r* can be off by up to 5bp. See notes/DECISIONS.md (2026-09-27).

Any other published r* reads the same way. HLW's real-time vintages (the week 9
robustness r*, ``rule.rstar: {source: nyfed, series: HLW_RSTAR, real: true}``)
are a log of the same shape, one value a quarter, held until the next; they are
already real rates, so ``real: true`` skips the target subtraction. Taylor's 2%
stands in before HLW's first vintage too (published 2016-03-05). See
notes/DECISIONS.md (V2).
"""

import numpy as np
import pandas as pd
from policypath.model.reaction import on

DAY = pd.Timedelta(days=1)


def rstar(sep, as_of, spec):
    """r* in percent at the end of `as_of`.

    `sep` has ``date``, ``value`` (the longer-run median, percent) and
    ``published``; `spec` is the rule block. Returns ``rstar``, the SEP it came
    from (``sep_date``, NaT before the first) and that SEP's ``longer_run`` median.

    A currency whose central bank publishes no longer-run rate gives
    ``rstar: {constant: x}`` instead, and `sep` is None. With ``real: true``
    the published value is r* itself (HLW), and ``longer_run`` is that value.
    """
    if "constant" in spec["rstar"]:
        return {"rstar": spec["rstar"]["constant"], "sep_date": pd.NaT, "longer_run": np.nan}
    known = sep[sep["published"] < pd.Timestamp(as_of).normalize() + DAY]
    if known.empty:
        return {"rstar": spec["rstar"]["before_first"], "sep_date": pd.NaT, "longer_run": np.nan}
    last = known.sort_values(["date", "published"]).iloc[-1]
    value = last["value"] if spec["rstar"].get("real", False) else last["value"] - on(spec["inflation_target"], as_of)
    return {"rstar": value, "sep_date": last["date"], "longer_run": last["value"]}
