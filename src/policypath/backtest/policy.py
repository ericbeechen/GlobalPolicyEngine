"""The policy-gap z through the engine: the one expression this week trades."""

from policypath.backtest import engine
from policypath.backtest.instrument import held_rate_change


def run(signal, sessions, meetings, k, spec):
    """P&L from trading the rate after the k-th meeting ahead on the k-th gap z.

    `signal` is `signal.gap.build`'s long frame; `spec` the config's backtest block.
    """
    z = signal[signal["k"] == k].set_index("session")["z"]
    change_bp = held_rate_change(sessions, meetings, k) * 100.0
    return engine.run(z, change_bp, spec["dv01"], spec["execution_lag"])
