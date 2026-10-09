"""Recognising a run's token-budget stop without importing the host's code.

The platform's worker caps tokens per run: its callback raises
``RunBudgetExceeded`` once a run passes its cap. The fork never imports the
worker, so it recognises that error by its class name (anywhere in the MRO)
or by a ``run_budget_exceeded`` marker attribute. A budget stop must never be
retried or swallowed by a best-effort ``except Exception``: every retry spends
more tokens past the cap, and a swallowed stop lets the run go on spending.
"""

from __future__ import annotations

BUDGET_ERROR_NAMES = frozenset({"RunBudgetExceeded"})


def is_budget_exceeded(exc: BaseException | None) -> bool:
    """True when ``exc`` (or an exception it was raised from) is a budget stop."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "run_budget_exceeded", False):
            return True
        if any(cls.__name__ in BUDGET_ERROR_NAMES for cls in type(exc).__mro__):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def reraise_if_budget(exc: BaseException) -> None:
    """Call first in a broad ``except``: re-raises a budget stop unchanged."""
    if is_budget_exceeded(exc):
        raise exc
