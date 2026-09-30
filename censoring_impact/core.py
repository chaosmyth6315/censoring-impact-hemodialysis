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


_TRUE, _FALSE = {'1', 'true', 'yes'}, {'0', 'false', 'no'}


def binary(values, name: str) -> np.ndarray:
    """0/1 flags as booleans; anything else (strings like 'maybe', missing values, 2) is an error."""
    v = np.asarray(values)
    if v.dtype == bool:
        return v
    if v.dtype.kind in 'iuf':
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f'{name} must be 0/1; found {sorted(set(v[~np.isin(v, (0, 1))].tolist()))[:5]}')
        return v.astype(bool)
    s = pd.Series(v).astype(str).str.strip().str.lower()
    bad = ~s.isin(_TRUE | _FALSE) | pd.isna(pd.Series(v))
    if bad.any():
        raise ValueError(f'{name} must be 0/1 or true/false; found {sorted(pd.Series(v)[bad].astype(str).unique())[:5]}')
    return s.isin(_TRUE).to_numpy()


def fraction_name(f: float) -> str:
    """Scenario identifier for an assumed share: two to four decimals, so distinct shares never share a name."""
    s = f'{f:.4f}'.rstrip('0')
    return 'fraction_' + (s if len(s.split('.')[1]) >= 2 else f'{f:.2f}')


def check_fractions(fractions) -> tuple:
    fr = tuple(float(f) for f in fractions)
    for f in fr:
        if not 0 < f < 1:
            raise ValueError(f'assumed shares must be between 0 and 1, got {f}')
        if abs(f * 1e4 - round(f * 1e4)) > 1e-6:
            raise ValueError(f'assumed shares take at most four decimal places, got {f}')
    names = [fraction_name(f) for f in fr]
    if len(set(names)) != len(names):
        raise ValueError(f'assumed shares must be distinct, got {fr}')
    return fr


@dataclass
class Arm:
    """Rows of one data set (the evaluation set, or the set used to estimate censoring weights)."""
    pid: np.ndarray          # cluster (patient) identifier per row
    time: np.ndarray         # follow-up time per row
    event: np.ndarray        # death observed, as analyzed (bool)
    departed: np.ndarray     # the row's patient left follow-up (bool, constant within patient)
    route: np.ndarray | None = None   # route code per row (constant within patient), or None

    def __post_init__(self):
        raw = pd.Series(np.asarray(self.pid, dtype=object))
        if raw.isna().any() or (raw.astype(str).str.strip() == '').any():
            raise ValueError('every row needs a patient identifier')
        self.pid = raw.astype(str).to_numpy()
        self.time = np.asarray(self.time, dtype=float)
        self.event = binary(self.event, 'event')
        self.departed = binary(self.departed, 'departed')
        n = len(self.pid)
        if not (len(self.time) == len(self.event) == len(self.departed) == n):
            raise ValueError('id, time, event and departed must have the same length')
        if not np.isfinite(self.time).all() or (self.time < 0).any():
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
    if stay.empty:
        raise ValueError('every patient left follow-up, so the stayer rate is undefined')
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
    fr = [Scenario(fraction_name(f), 'fraction', float(f)) for f in check_fractions(fractions)]
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
        for name, frac in [(fraction_name(f), f) for f in fractions] + [('stayer_rate', rate)]:
            k = int(round(frac * len(ids)))            # halves round to even (the web version matches)
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
    of predicted risk, with inverse-probability-of-censoring weights from the censoring estimator cde. O:E is
    always returned; slope and intercept are NaN when they cannot be estimated (one outcome class only, unusable
    weights, or no spread in predicted risk).
    """
    t, e = time[idx], event[idx]
    kt, ks = kaplan_meier_estimator(e, t)
    observed = float(1 - (ks[kt <= horizon][-1] if np.any(kt <= horizon) else 1.0))
    b = np.full(t.size, -1, dtype=int)
    hit, free = e & (t <= horizon), t > horizon
    b[hit], b[free] = 1, 0
    ok = b >= 0
    w = np.zeros(t.size)
    fit_ok = np.unique(b[ok]).size == 2
    if fit_ok:
        if hit.any():
            w[hit] = 1.0 / cde.predict_proba(t[hit])
        if free.any():
            w[free] = 1.0 / cde.predict_proba(np.full(int(free.sum()), horizon))
        fit_ok = bool(np.isfinite(w[ok]).all() and (w[ok] > 0).all())
    out = {}
    for m, risk in risk_h.items():
        r = risk[idx]
        expected = float(r.mean())
        slope = intercept = float('nan')
        x = np.log(np.clip(r[ok], 1e-6, 1 - 1e-6) / np.clip(1 - r[ok], 1e-6, 1))
        if fit_ok and np.ptp(x) > 0:
            f = LogisticRegression(solver='lbfgs', max_iter=1000, **_UNPENALIZED).fit(x[:, None], b[ok], sample_weight=w[ok])
            slope, intercept = float(f.coef_[0, 0]), float(f.intercept_[0])
        out[m] = dict(slope=slope, intercept=intercept, observed=observed, expected=expected,
                      oe=observed / expected if expected > 0 else float('nan'), n_km=len(idx),
                      n_slope=int(ok.sum()) if np.isfinite(slope) else 0)
    return out


def oe_crossing(points) -> dict | None:
    """
    Assumed share at which O:E crosses 1, from (share, O:E) pairs in order of share: an exact 1 if one is observed,
    otherwise linear interpolation across the first pair of neighbours on opposite sides of 1.
    """
    pts = sorted(points)
    for f, oe in pts:
        if oe == 1:
            return dict(crossing_lower_fraction=f, crossing_lower_oe=oe, crossing_upper_fraction=f,
                        crossing_upper_oe=oe, interpolated_crossing=f)
    for (f0, o0), (f1, o1) in zip(pts, pts[1:]):
        if (o0 - 1) * (o1 - 1) < 0 and f1 > f0:
            cross = f0 + (1 - o0) * (f1 - f0) / (o1 - o0)
            return dict(crossing_lower_fraction=f0, crossing_lower_oe=o0,
                        crossing_upper_fraction=f1, crossing_upper_oe=o1, interpolated_crossing=cross)
    return None
