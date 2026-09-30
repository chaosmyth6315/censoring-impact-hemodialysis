"""
Command line:

    python -m censoring_impact --data evaluation.csv --id patient_id --time time --event event \
        --departed departed --model "Cox=risk_cox" --model "Forest=risk_rsf" --reference Cox \
        [--risk-at-horizon "Cox=risk3y_cox" --risk-at-horizon "Forest=risk3y_rsf" --horizon 1095.75] \
        [--calibration-eligible horizon_observable] [--route route] [--weights-data development.csv] \
        [--between-patient] --out results/

One row per evaluation unit (a patient, or a patient-year landmark). Patients can contribute several rows; the
bootstrap resamples patients. See README.md for the column definitions.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis import Config, run
from .core import Arm
from .report import write


def _pairs(values, what):
    out = {}
    for v in values or []:
        if '=' not in v:
            raise SystemExit(f'{what} must be given as NAME=COLUMN, got {v!r}')
        k, c = v.split('=', 1)
        out[k.strip()] = c.strip()
    return out


def _bool(s: pd.Series, name: str) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    m = {'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False}
    v = s.astype(str).str.strip().str.lower().map(m)
    if v.isna().any():
        raise SystemExit(f'column {name!r} must be 0/1 or true/false; found {sorted(s[v.isna()].astype(str).unique())[:5]}')
    return v.to_numpy(dtype=bool)


def _read(path: str, a) -> pd.DataFrame:
    """Read a CSV, keeping the identifier (and route) exactly as written: '001' and '1' stay different patients."""
    text_cols = [c for c in (a.id, a.route) if c]
    df = pd.read_csv(path, dtype={c: str for c in text_cols}, keep_default_na=False, na_values={})
    for c in df.columns:
        if c not in text_cols:
            df[c] = df[c].replace('', np.nan) if df[c].dtype == object else df[c]
    return df


def _num(df: pd.DataFrame, c: str, where: str) -> np.ndarray:
    v = pd.to_numeric(df[c], errors='coerce')
    bad = v.isna() & df[c].notna() | df[c].isna()
    if bad.any():
        raise SystemExit(f'{where}: column {c!r} must be numeric on every row; '
                         f'problem at data row(s) {list(np.flatnonzero(bad.to_numpy())[:5] + 2)}')
    return v.to_numpy(dtype=float)


def _risk_h(df: pd.DataFrame, c: str, elig: np.ndarray) -> np.ndarray:
    """Predicted risk by the horizon: must be numeric on calibration-eligible rows; other rows are not used."""
    _num(df[elig], c, '--data (calibration-eligible rows)')
    return pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)


def _arm(df: pd.DataFrame, a, label: str) -> Arm:
    need = [a.id, a.time, a.event, a.departed] + ([a.route] if a.route else [])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f'{label}: missing columns {miss}')
    return Arm(pid=df[a.id], time=_num(df, a.time, label), event=_bool(df[a.event], a.event),
               departed=_bool(df[a.departed], a.departed), route=df[a.route] if a.route else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog='censoring_impact', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='evaluation data (CSV)')
    p.add_argument('--id', required=True, help='patient (cluster) identifier column')
    p.add_argument('--time', required=True, help='follow-up time column (any unit; same unit as --horizon)')
    p.add_argument('--event', required=True, help='death observed column (0/1)')
    p.add_argument('--departed', required=True, help='patient left follow-up column (0/1, constant within patient)')
    p.add_argument('--model', action='append', required=True, help='NAME=COLUMN of a risk score (higher = higher risk)')
    p.add_argument('--reference', required=True, help='model NAME that differences are taken against')
    p.add_argument('--risk-at-horizon', action='append', help='NAME=COLUMN of predicted risk of death by the horizon (0-1)')
    p.add_argument('--horizon', type=float, help='calibration horizon, in the unit of --time')
    p.add_argument('--calibration-eligible', help='0/1 column: rows whose horizon is fully observable (default: all rows)')
    p.add_argument('--route', help="route column for departed patients: 'censored', 'dead' or 'random'")
    p.add_argument('--weights-data', help='separate data (same column names) for estimating censoring weights, '
                                          'e.g. the development set; default: the evaluation data')
    p.add_argument('--fractions', default='0.25,0.50,0.75', help='assumed shares of departed patients who died')
    p.add_argument('--draws', type=int, default=200, help='random assignments per random scenario')
    p.add_argument('--boot', type=int, default=1000, help='bootstrap replicates')
    p.add_argument('--seed', type=int, default=20260903)
    p.add_argument('--between-patient', action='store_true', help='also compute C from between-patient pairs only')
    p.add_argument('--out', required=True, help='output folder')
    a = p.parse_args(argv)

    df = _read(a.data, a)
    ev = _arm(df, a, '--data')
    models = _pairs(a.model, '--model')
    rh = _pairs(a.risk_at_horizon, '--risk-at-horizon') or None
    for c in list(models.values()) + list((rh or {}).values()):
        if c not in df.columns:
            raise SystemExit(f'--data: missing column {c!r}')
    weights = _arm(_read(a.weights_data, a), a, '--weights-data') if a.weights_data else None
    elig = _bool(df[a.calibration_eligible], a.calibration_eligible) if a.calibration_eligible else np.ones(len(df), bool)
    cfg = Config(risk={k: _num(df, c, '--data') for k, c in models.items()}, reference=a.reference,
                 risk_h={k: _risk_h(df, c, elig) for k, c in rh.items()} if rh else None,
                 horizon=a.horizon,
                 calibration_eligible=elig if a.calibration_eligible else None,
                 fractions=tuple(float(x) for x in a.fractions.split(',')), n_draws=a.draws, n_boot=a.boot,
                 seed=a.seed, between_patient=a.between_patient)
    try:
        res = run(ev, cfg, weights)
    except ValueError as e:
        raise SystemExit(f'error: {e}')
    record = dict(tool='censoring_impact', version=__version__, python=platform.python_version(),
                  arguments=vars(a), command=' '.join(sys.argv))
    files = write(res, cfg, Path(a.out), record)
    print('\nWritten:'); [print('  ', f) for f in files]


if __name__ == '__main__':
    main()
