"""
Runs the censoring-impact analysis end to end: point estimates in every scenario, patient-clustered bootstrap
intervals in the key scenarios, and the observed-to-expected crossing point.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import (Arm, Assignments, Scenario, between_patient_c, build_scenarios, calibration,
                   censoring_estimator, generate_draws, harrell_c, oe_crossing, stayer_rate)


@dataclass
class Config:
    risk: dict                       # model name -> risk score per evaluation row (higher = higher risk)
    reference: str                   # model that differences are taken against
    risk_h: dict | None = None       # model name -> predicted risk of death by the horizon (enables calibration)
    horizon: float | None = None
    calibration_eligible: np.ndarray | None = None   # rows whose horizon is fully observable (default: all)
    fractions: tuple = (0.25, 0.50, 0.75)
    n_draws: int = 200
    n_boot: int = 1000
    seed: int = 20260903
    between_patient: bool = False
    min_events_boot: int = 20         # skip a bootstrap C when fewer events than this
    min_calibration_rows: int = 200   # skip a calibration replicate with fewer eligible rows than this


@dataclass
class Results:
    exposure: dict
    scenarios: list
    draws: pd.DataFrame
    discrimination: pd.DataFrame
    paired: pd.DataFrame
    calibration: pd.DataFrame | None = None
    crossing: dict | None = None
    notes: list = field(default_factory=list)


def _key_scenarios(scenarios):
    order = ['as_analyzed', 'stayer_rate', 'route_informed', 'all_departed_died']
    by = {s.name: s for s in scenarios}
    return [by[n] for n in order if n in by]


def run(evaluation: Arm, cfg: Config, weights: Arm | None = None, log=print) -> Results:
    if cfg.reference not in cfg.risk:
        raise ValueError(f'reference model {cfg.reference!r} is not among the models {list(cfg.risk)}')
    models = list(cfg.risk)
    risk = {m: np.asarray(v, dtype=float) for m, v in cfg.risk.items()}
    n = len(evaluation.pid)
    for m, v in risk.items():
        if len(v) != n or np.isnan(v).any():
            raise ValueError(f'risk scores for {m!r} must be complete and match the data')
    with_route = evaluation.route is not None
    arms = {'evaluation': evaluation}
    if weights is not None:
        overlap = set(evaluation.departed_ids) & set(weights.departed_ids)
        if overlap:
            raise ValueError('departed patients must not appear in both the evaluation and the weights data')
        if with_route and weights.route is None:
            raise ValueError('the weights data need the route column too')
        arms['weights'] = weights
    rate = stayer_rate(evaluation)
    scenarios = build_scenarios(cfg.fractions, rate, with_route)
    draws = generate_draws(arms, cfg.fractions, rate, cfg.n_draws, cfg.seed, with_route)
    A = Assignments(arms, draws, cfg.n_draws)
    ids = evaluation.departed_ids
    exposure = dict(
        rows=n, patients=int(np.unique(evaluation.pid).size),
        rows_from_departed_patients=int(evaluation.departed.sum()),
        share_rows_departed=float(evaluation.departed.mean()), departed_patients=int(ids.size),
        events_recorded_in_departed_rows=int(evaluation.event[evaluation.departed].sum()),
        events_as_analyzed=int(evaluation.event.sum()),
        patients_died_as_analyzed=int(pd.Series(evaluation.event).groupby(evaluation.pid).max().sum()),
        stayer_rate=rate, departed_assigned_at_stayer_rate=int(round(rate * ids.size)),
        route_informed=with_route, weights_data='separate' if weights is not None else 'evaluation data')
    t0 = _time.time()

    # ---------------------------------------------------------- discrimination: point estimates
    log('Discrimination in every scenario …')
    rows = []
    for sc in scenarios:
        nd = A.draws_for(sc)
        acc = {m: [] for m in models}; nev, ndead = [], []
        for d in range(nd):
            ev = A.events(sc, d)
            for m in models:
                acc[m].append(harrell_c(ev, evaluation.time, risk[m]))
            nev.append(int(ev.sum())); ndead.append(len(A.dead_ids(sc, d, 'evaluation')))
        c = {m: float(np.mean(acc[m])) for m in models}
        rec = dict(scenario=sc.name, assumed_fraction=sc.fraction, n_draws=nd,
                   departed_assigned_death=float(np.mean(ndead)), events=float(np.mean(nev)),
                   event_rate=float(np.mean(nev)) / n, best_model=max(c, key=c.get),
                   **{f'c_{m}': c[m] for m in models})
        if cfg.between_patient and sc.name in ('as_analyzed', 'stayer_rate', 'all_departed_died'):
            bx = [between_patient_c(A.events(sc, d), evaluation.time, evaluation.pid, risk) for d in range(nd)]
            rec.update({f'cx_{m}': float(np.mean([x[m] for x in bx])) for m in models})
        rows.append(rec)
    disc = pd.DataFrame(rows)

    # ---------------------------------------------------------- discrimination: paired bootstrap
    key = _key_scenarios(scenarios)
    groups = {p: np.flatnonzero(evaluation.pid == p) for p in np.unique(evaluation.pid)}
    patients = np.array(sorted(groups))

    def cw(idx, ev):
        e = ev[idx]
        if e.sum() < cfg.min_events_boot:
            return None
        try:
            return {m: harrell_c(e, evaluation.time[idx], risk[m][idx]) for m in models}
        except Exception:
            return None

    log(f'Paired bootstrap, {cfg.n_boot} patient-clustered replicates …')
    rng_pt, rng_dr = np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed + 1)
    boot = {s.name: [] for s in key}
    for b in range(cfg.n_boot):
        pick = rng_pt.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        for sc in key:
            d = 0 if sc.deterministic else int(rng_dr.integers(cfg.n_draws))
            r = cw(idx, A.events(sc, d))
            if r is not None:
                boot[sc.name].append(r)
        if (b + 1) % 250 == 0:
            log(f'  {b + 1}/{cfg.n_boot} ({_time.time() - t0:.0f}s)')
    dsc = disc.set_index('scenario')
    ALL = np.arange(n)
    prow = []
    for sc in key:
        for m in models:
            if m == cfg.reference:
                continue
            diffs = np.array([x[m] - x[cfg.reference] for x in boot[sc.name]])
            lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs.size else (np.nan, np.nan)
            sd = np.nan
            if not sc.deterministic:
                v = [cw(ALL, A.events(sc, d)) for d in range(cfg.n_draws)]
                sd = float(np.std([x[m] - x[cfg.reference] for x in v if x is not None], ddof=1))
            prow.append(dict(scenario=sc.name, model=m, reference=cfg.reference,
                             c_model=float(dsc.loc[sc.name, f'c_{m}']), c_reference=float(dsc.loc[sc.name, f'c_{cfg.reference}']),
                             delta=float(dsc.loc[sc.name, f'c_{m}'] - dsc.loc[sc.name, f'c_{cfg.reference}']),
                             ci_low=float(lo), ci_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0),
                             n_boot=int(diffs.size), assignment_sd=sd))
    paired = pd.DataFrame(prow)
    res = Results(exposure=exposure, scenarios=scenarios, draws=draws, discrimination=disc, paired=paired)

    # ---------------------------------------------------------- calibration
    if cfg.risk_h is None:
        res.notes.append('Calibration not computed: no predicted risks at a horizon were supplied.')
        return res
    if cfg.horizon is None:
        raise ValueError('a horizon is needed for calibration')
    risk_h = {m: np.asarray(v, dtype=float) for m, v in cfg.risk_h.items()}
    elig = (np.ones(n, dtype=bool) if cfg.calibration_eligible is None
            else np.asarray(cfg.calibration_eligible).astype(bool))
    sub = np.flatnonzero(elig)
    warm = 'weights' if weights is not None else 'evaluation'
    wa = arms[warm]

    def cde_for(sc, d=0):
        return censoring_estimator(A.events(sc, d, arm=warm), wa.time)

    log('Calibration in every scenario …')
    crow = []
    for sc in scenarios:
        nd = A.draws_for(sc)
        acc = {m: {k: [] for k in ('slope', 'intercept', 'observed', 'expected', 'oe', 'n_km', 'n_slope')} for m in risk_h}
        for d in range(nd):
            r = calibration(A.events(sc, d), evaluation.time, sub, risk_h, cfg.horizon, cde_for(sc, d))
            if r is None:
                continue
            for m in risk_h:
                for k in acc[m]:
                    acc[m][k].append(r[m][k])
        for m in risk_h:
            rec = {k: (float(np.mean(v)) if v else np.nan) for k, v in acc[m].items()}
            rec['n_km'] = int(round(rec['n_km'])) if acc[m]['n_km'] else 0
            rec['n_slope'] = int(round(rec['n_slope'])) if acc[m]['n_slope'] else 0
            crow.append(dict(scenario=sc.name, assumed_fraction=sc.fraction, model=m, n_draws=nd, **rec))
    cal = pd.DataFrame(crow)

    log(f'Calibration bootstrap, {cfg.n_boot} patient-clustered replicates …')
    rng_pt, rng_dr = np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed + 1)
    cde = {sc.name: [cde_for(sc, d) for d in range(A.draws_for(sc))] for sc in key}
    bacc = {(sc.name, m, q): [] for sc in key for m in risk_h for q in ('slope', 'oe')}
    for b in range(cfg.n_boot):
        pk = rng_pt.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in pk])
        idx = idx[elig[idx]]
        if len(idx) < cfg.min_calibration_rows:
            continue
        for sc in key:
            d = 0 if sc.deterministic else int(rng_dr.integers(cfg.n_draws))
            r = calibration(A.events(sc, d), evaluation.time, idx, risk_h, cfg.horizon, cde[sc.name][d])
            if r is None:
                continue
            for m in risk_h:
                bacc[(sc.name, m, 'slope')].append(r[m]['slope']); bacc[(sc.name, m, 'oe')].append(r[m]['oe'])
        if (b + 1) % 250 == 0:
            log(f'  {b + 1}/{cfg.n_boot} ({_time.time() - t0:.0f}s)')
    for i, r in cal.iterrows():
        if r.scenario not in {s.name for s in key}:
            continue
        for q in ('slope', 'oe'):
            v = bacc[(r.scenario, r.model, q)]
            if v:
                cal.loc[i, f'{q}_ci_low'], cal.loc[i, f'{q}_ci_high'] = np.percentile(v, [2.5, 97.5])
            cal.loc[i, f'{q}_n_boot'] = len(v)
    res.calibration = cal
    ref = cal[cal.model == cfg.reference] if cfg.reference in risk_h else cal[cal.model == list(risk_h)[0]]
    pts = [(float(r.assumed_fraction), float(r.oe)) for r in ref.itertuples()
           if r.assumed_fraction is not None and not pd.isna(r.assumed_fraction) and r.scenario != 'route_informed'
           and np.isfinite(r.oe)]
    res.crossing = oe_crossing(pts)
    log(f'Done in {_time.time() - t0:.0f}s')
    return res
