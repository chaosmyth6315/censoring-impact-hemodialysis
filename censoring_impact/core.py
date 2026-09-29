"""
Core computations for the censoring-impact analysis.

The fitted models are held fixed. Only the outcome definition changes: a chosen share of the patients who left
follow-up ("departed" patients) is reclassified as having died at the moment of departure, with follow-up time
unchanged, and discrimination and calibration are recomputed. Everything here mirrors the analysis in
Fang et al., "How much is the non-informative censoring assumption worth?", so that the published numbers are
reproduced exactly from the same inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.util import Surv

ROUTE_CODES = ('censored', 'dead', 'random')   # per departed patient, used by the route-informed scenario
TIE_TOL = 1e-8                                  # risk ties, as in sksurv.concordance_index_censored

_SK = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
# An unpenalized logistic regression: `penalty=None` before scikit-learn 1.8, `C=inf` from 1.8 on (same fit).
_UNPENALIZED = dict(C=np.inf) if _SK >= (1, 8) else dict(penalty=None)


@dataclass
class Arm:
    """Rows of one data set (the evaluation set, or the set used to estimate censoring weights)."""
    pid: np.ndarray          # cluster (patient) identifier per row
    time: np.ndarray         # follow-up time per row
    event: np.ndarray        # death observed, as analyzed (bool)
    departed: np.ndarray     # the row's patient left follow-up (bool, constant within patient)
    route: np.ndarray | None = None   # route code per row (constant within patient), or None

    def __post_init__(self):
        self.pid = np.asarray(self.pid).astype(str)
        self.time = np.asarray(self.time, dtype=float)
        self.event = np.asarray(self.event).astype(bool)
        self.departed = np.asarray(self.departed).astype(bool)
        n = len(self.pid)
        if not (len(self.time) == len(self.event) == len(self.departed) == n):
            raise ValueError('id, time, event and departed must have the same length')
        if np.isnan(self.time).any() or (self.time < 0).any():
            raise ValueError('follow-up times must be non-negative numbers')
        per_pt = pd.Series(self.departed).groupby(self.pid).nunique()
        if (per_pt > 1).any():
            raise ValueError(f'departed flag varies within patient: {list(per_pt[per_pt > 1].index[:5])}')
        if self.route is not None:
            self.route = np.asarray(self.route).astype(str)
            bad = set(self.route[self.departed]) - set(ROUTE_CODES)
            if bad:
                raise ValueError(f'route codes for departed patients must be {ROUTE_CODES}; found {sorted(bad)}')
            if (pd.Series(self.route).groupby(self.pid).nunique() > 1).any():
                raise ValueError('route code varies within patient')

    @property
    def departed_ids(self) -> np.ndarray:
        return np.array(sorted(set(self.pid[self.departed])))

    def route_of(self) -> dict:
        return dict(zip(self.pid, self.route)) if self.route is not None else {}


def stayer_rate(arm: Arm) -> float:
    """Share of never-departed patients with a recorded death (patient level)."""
    stay = pd.Series(arm.event[~arm.departed]).groupby(arm.pid[~arm.departed]).max()
    return float(stay.mean())


@dataclass
class Scenario:
    name: str
    kind: str                # 'none' | 'fraction' | 'stayer' | 'route' | 'all'
    fraction: float | None   # assumed share of departed patients who died (None for route)

    @property
    def deterministic(self) -> bool:
        return self.kind in ('none', 'all')


def build_scenarios(fractions, rate, with_route) -> list[Scenario]:
    """Scenarios in the order reported: none, the fractions and the stayer rate by size, route-informed, all."""
    fr = [Scenario(f'fraction_{f:.2f}', 'fraction', float(f)) for f in fractions]
    fr.append(Scenario('stayer_rate', 'stayer', rate))
    fr.sort(key=lambda s: s.fraction)
    out = [Scenario('as_analyzed', 'none', 0.0)] + fr
    if with_route:
        out.append(Scenario('route_informed', 'route', None))
    out.append(Scenario('all_departed_died', 'all', 1.0))
    return out


def generate_draws(arms: dict[str, Arm], fractions, rate, n_draws, seed, with_route) -> pd.DataFrame:
    """
    Random assignments of death among departed patients, generated once and shared by every analysis.

    One generator, seeded once, is used for all arms in turn (evaluation arm first), each arm drawing the
    fractions in the order given, then the stayer rate, then the route-informed scenario. The same share is
    applied to every arm, because a scenario is an assumption about the world, but each arm has its own
    departed patients and therefore its own draws.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for arm_name, arm in arms.items():
        ids = arm.departed_ids
        pos = {p: i for i, p in enumerate(ids)}
        for name, frac in [(f'fraction_{f:.2f}', f) for f in fractions] + [('stayer_rate', rate)]:
            k = int(round(frac * len(ids)))
            for d in range(n_draws):
                mask = np.zeros(len(ids), dtype=int)
                mask[[pos[p] for p in rng.choice(ids, size=k, replace=False)]] = 1
                rows.append(dict(arm=arm_name, scenario=name, draw=d, mask=''.join(map(str, mask))))
        if with_route:
            code = arm.route_of()
            stoch = np.array([p for p in ids if code[p] == 'random'])
            forced = [p for p in ids if code[p] == 'dead']
            k = int(round(rate * len(stoch)))
            for d in range(n_draws):
                mask = np.zeros(len(ids), dtype=int)
                pick = set(rng.choice(stoch, size=k, replace=False)) | set(forced)
                mask[[pos[p] for p in pick]] = 1
                rows.append(dict(arm=arm_name, scenario='route_informed', draw=d, mask=''.join(map(str, mask))))
    return pd.DataFrame(rows)


class Assignments:
    """Look-up of scenario events per arm and draw."""

    def __init__(self, arms: dict[str, Arm], draws: pd.DataFrame, n_draws: int):
        self.arms, self.n_draws = arms, n_draws
        self._mask = {(r.arm, r.scenario, r.draw): np.array(list(r.mask), dtype=int).astype(bool)
                      for r in draws.itertuples()}

    def draws_for(self, sc: Scenario) -> int:
        return 1 if sc.deterministic else self.n_draws

    def dead_ids(self, sc: Scenario, draw: int, arm: str) -> np.ndarray:
        ids = self.arms[arm].departed_ids
        if sc.kind == 'none':
            return ids[:0]
        if sc.kind == 'all':
            return ids
        return ids[self._mask[(arm, sc.name, draw)]]

    def events(self, sc: Scenario, draw: int = 0, arm: str = 'evaluation') -> np.ndarray:
        a = self.arms[arm]
        return a.event | (a.departed & np.isin(a.pid, self.dead_ids(sc, draw, arm)))


# ---------------------------------------------------------------- discrimination
def harrell_c(event, time, risk) -> float:
    return float(concordance_index_censored(event, time, risk)[0])


def between_patient_c(event, time, pid, risks: dict, chunk: int = 1000) -> dict:
    """
    Harrell's C restricted to pairs formed between different patients.

    Comparable pairs follow sksurv exactly: the earlier time is an event, or at the same time an event is paired
    with a censored observation. Risk ties within 1e-8 count one half. Computed in row chunks to bound memory.
    """
    n = len(time)
    num = {m: 0.0 for m in risks}
    den = 0
    for s in range(0, n, chunk):
        e = slice(s, min(s + chunk, n))
        ti, tj = time[e][:, None], time[None, :]
        comp = event[e][:, None] & ((ti < tj) | ((ti == tj) & ~event[None, :])) & (pid[e][:, None] != pid[None, :])
        den += int(comp.sum())
        for m, r in risks.items():
            d = r[e][:, None] - r[None, :]
            num[m] += int(((d > TIE_TOL) & comp).sum()) + 0.5 * int(((np.abs(d) <= TIE_TOL) & comp).sum())
    return {m: (num[m] / den if den else float('nan')) for m in risks}


# ---------------------------------------------------------------- calibration
def censoring_estimator(event, time):
    return CensoringDistributionEstimator().fit(Surv.from_arrays(event, time))


def calibration(event, time, idx, risk_h: dict, horizon: float, cde):
    """
    Calibration at the horizon on rows idx.

    Observed-to-expected: Kaplan-Meier risk at the horizon over all rows idx, divided by the mean predicted risk.
    Slope: logistic regression of known horizon status (died by the horizon, or followed beyond it) on the logit
    of predicted risk, with inverse-probability-of-censoring weights from the censoring estimator cde.
    Returns None if the slope cannot be estimated.
    """
    t, e = time[idx], event[idx]
    b = np.full(t.size, -1, dtype=int)
    hit, free = e & (t <= horizon), t > horizon
    b[hit], b[free] = 1, 0
    w = np.zeros(t.size)
    if hit.any():
        w[hit] = 1.0 / cde.predict_proba(t[hit])
    if free.any():
        w[free] = 1.0 / cde.predict_proba(np.full(int(free.sum()), horizon))
    ok = b >= 0
    if np.unique(b[ok]).size < 2 or not np.isfinite(w[ok]).all() or (w[ok] <= 0).any():
        return None
    kt, ks = kaplan_meier_estimator(event[idx], time[idx])
    observed = float(1 - (ks[kt <= horizon][-1] if np.any(kt <= horizon) else 1.0))
    out = {}
    for m, risk in risk_h.items():
        r = risk[idx]
        x = np.log(np.clip(r[ok], 1e-6, 1 - 1e-6) / np.clip(1 - r[ok], 1e-6, 1))
        f = LogisticRegression(solver='lbfgs', max_iter=1000, **_UNPENALIZED).fit(x[:, None], b[ok], sample_weight=w[ok])
        expected = float(r.mean())
        out[m] = dict(slope=float(f.coef_[0, 0]), intercept=float(f.intercept_[0]), observed=observed,
                      expected=expected, oe=observed / expected, n_km=len(idx), n_slope=int(ok.sum()))
    return out


def oe_crossing(points) -> dict | None:
    """Linear interpolation of the assumed fraction at which O:E crosses 1, from (fraction, O:E) pairs."""
    pts = sorted(points)
    below = [p for p in pts if p[1] < 1]
    above = [p for p in pts if p[1] > 1]
    if not below or not above or above[0][0] <= below[-1][0]:
        return None
    lo, hi = below[-1], above[0]
    cross = lo[0] + (1 - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])
    return dict(crossing_lower_fraction=lo[0], crossing_lower_oe=lo[1],
                crossing_upper_fraction=hi[0], crossing_upper_oe=hi[1], interpolated_crossing=cross)
