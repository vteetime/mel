"""Config loading + position sizing.

Sizing is the one place Tee's risk rule lives, so it is a pure function and is unit
tested. Rule: margin deployed per position <= max_margin_fraction of available funds.
At leverage L the notional is margin * L.

── CONTRACT SIZE, AND WHY KRAKEN IS NOT KUCOIN ────────────────────────────────
`contracts` is an exchange quantity, never a coin count. The two venues Mel's
ancestry spans disagree about what one contract IS:

    KuCoin ETHUSDTM : contract_size 0.01  -> amounts are WHOLE contracts
                      (3 contracts = 0.03 ETH). Jack floors to int.
    Kraken  ETH/USD:USD : contract_size 1.0 -> the amount IS ETH, quoted to
                      3 decimals (0.138 = 0.138 ETH), so it is NOT a whole number.

Both money figures go through `contracts * contract_size`:

    base_asset_qty = contracts * contract_size
    notional       = base_asset_qty * price

Getting this wrong is a 100x error in one direction (ignoring contract_size on a
0.01-contract venue) and a silent no-trade in the other (flooring Kraken's 0.138 ETH
to 0 contracts, after which every tick looks healthy and nothing ever trades).
Mel therefore floors to the market's AMOUNT STEP — passed in from the exchange, not
hardcoded — and refuses below `min_contracts`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


@dataclass
class Sizing:
    leverage: int
    margin: float                 # quote currency of margin committed
    notional: float               # margin * leverage
    contracts: float              # exchange contracts (Kraken: this IS the ETH amount)
    contract_size: float
    scale: float = 1.0            # confidence / drawdown multiplier actually applied
    contract_step: float = 1.0    # amount step actually applied
    min_contracts: float = 0.0    # floor below which we refuse to trade
    funds: float = 0.0            # free collateral this size was computed against
    margin_fraction: float = 0.0  # margin / funds actually deployed (the guard unit)


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def round_step(value: float, step: float, mode: str = "down") -> float:
    """Round to an exchange step. Guards against float dust making a 0.0 step.

    A non-positive or missing step means the venue gave us nothing usable, and the
    honest response is to leave the value alone rather than silently round to 0.
    """
    value = float(value)
    try:
        step = float(step)
    except (TypeError, ValueError):
        return value
    if not step or step <= 0:
        return value
    n = value / step
    # Floor with a tolerance: 0.138/0.001 must not land on 137.99999999 -> 0.137.
    n = math.floor(n + 1e-9) if mode == "down" else round(n)
    return float(round(n * step, 10))


COLLATERAL_CURRENCIES = ("USDT", "USD")


def available_funds(balance: dict, quote: str = "USDT") -> float:
    """Free collateral available to size a NEW position, summed over the wallet's currencies.

    Uses `free`, not `total`: margin already committed to an open position is not
    available to size a new one, and sizing off `total` would double-count it.

    [MONEY] Kraken futures runs a MULTI-COLLATERAL wallet: the same account can hold USD
    and USDT at the same time, and the venue converts them automatically when margin is
    required. Verified live 2026-11-11 on the Mel account: total USD 61.88 AND total USDT
    545.54, with the open XRP position's 431.35 of used margin appearing ONLY under USDT —
    so the two keys are separate ledgers, not two labels for one balance.

    The previous version returned the FIRST matching key, so sizing saw only the USDT row
    and silently ignored the USD collateral: 90% of 545.54 instead of 90% of 607.41, i.e.
    81% of the real account instead of the tested 90%. Sum across collateral currencies.

    Cross-currency is summed at par. USD/USDT trade within a few bps and the 90% rule
    leaves 10% of headroom, so par is far inside the tolerance; a rate call would add a
    network dependency to the entry path for no measurable gain.

    Falls back to `total` only when the venue reports no usable `free` map at all.
    """
    if not balance:
        return 0.0
    for src in (balance.get("free") or {}, balance.get("total") or {}):
        if not src:
            continue
        found = False
        total_free = 0.0
        for key in (quote,) + tuple(c for c in COLLATERAL_CURRENCIES if c != quote):
            v = src.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv < 0:
                continue
            total_free += fv
            found = True
        if found:
            return total_free
    return 0.0


def confidence_scale(confidence: int, enabled: bool = True) -> float:
    """Trip's G7 sizing: a bare-threshold entry is traded SMALLER, not the same."""
    if not enabled:
        return 1.0
    if confidence < 5:
        return 0.0
    if confidence == 5:
        return 0.5
    if confidence <= 7:
        return 0.75
    return 1.0


def size_position(funds: float, price: float, cfg: dict,
                  contract_size: float, contract_step: float = 1.0,
                  confidence: int = 8, extra_scale: float = 1.0,
                  min_contracts: float | None = None) -> Sizing:
    """Margin <= max_margin_fraction of `funds`, then scaled by confidence/drawdown.

    Size = floor(notional / (price * contract_size) to the market's amount step).
    May return contracts == 0 when the position would be too small, which callers must
    treat as "do not trade", never as "trade the minimum".
    """
    s = cfg["sizing"]
    lev = int(s["leverage"])
    max_frac = float(s["max_margin_fraction"])
    min_margin = float(s.get("min_margin_usdt", 0.0))
    if min_contracts is None:
        min_contracts = float(s.get("min_contracts", 1.0))

    conf_scale = confidence_scale(confidence, bool(s.get("confidence_scaling", True)))
    scale = conf_scale * max(0.0, float(extra_scale))
    margin = max(0.0, funds) * max_frac * scale

    blank = Sizing(leverage=lev, margin=0.0, notional=0.0, contracts=0.0,
                   contract_size=contract_size, scale=scale,
                   contract_step=contract_step, min_contracts=min_contracts,
                   funds=float(funds))

    if margin < min_margin or price <= 0 or contract_size <= 0:
        return blank

    notional = margin * lev
    raw = notional / (price * contract_size)
    contracts = round_step(raw, contract_step, "down")
    if contracts <= 0 or contracts + 1e-12 < min_contracts:
        return blank

    actual_notional = contracts * contract_size * price
    actual_margin = actual_notional / lev
    # Rounding down must never push margin back above the cap.
    assert actual_margin <= funds * max_frac + 1e-9, "sizing exceeded the margin cap"
    return Sizing(leverage=lev, margin=actual_margin, notional=actual_notional,
                  contracts=contracts, contract_size=contract_size, scale=scale,
                  contract_step=contract_step, min_contracts=min_contracts,
                  funds=float(funds),
                  # The unit the equity caps are actually measured in. Stored on the
                  # trade so guard accounting can convert margin-ROE back to EQUITY %.
                  margin_fraction=(actual_margin / float(funds)) if funds else 0.0)
