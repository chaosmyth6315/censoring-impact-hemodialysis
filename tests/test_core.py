"""Fast unit tests on synthetic data. Run: python -m pytest tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censoring_impact.analysis import Config, run  # noqa: E402
from censoring_impact.core import (Arm, Assignments, between_patient_c, build_scenarios, calibration,  # noqa: E402
                                   censoring_estimator, fraction_name, generate_draws, harrell_c, oe_crossing,
                                   stayer_rate)


def toy(n_pt=120, rows_per=3, seed=1, route=False):
    rng = np.random.default_rng(seed)
    pid = np.repeat([f'p{i:03d}' for i in range(n_pt)], rows_per)
    dep_pt = rng.random(n_pt) < .25
    dep = np.repeat(dep_pt, rows_per)
    time = rng.integers(1, 8, size=pid.size).astype(float) * 365          # many exact ties
    event = (rng.random(pid.size) < .4) & ~dep
    r = None
    if route:
        codes = rng.choice(['censored', 'dead', 'random', 'random', 'random'], size=n_pt)
        r = np.repeat(np.where(dep_pt, codes, ''), rows_per)
    return Arm(pid=pid, time=time, event=event, departed=dep, route=r), rng


def test_stayer_rate_is_patient_level():
    a = Arm(pid=['a', 'a', 'b', 'c', 'c'], time=[1, 2, 1, 1, 2], event=[0, 1, 0, 0, 0], departed=[0, 0, 0, 1, 1])
    assert stayer_rate(a) == pytest.approx(0.5)          # a died, b did not; c departed


def test_departed_flag_must_be_constant_within_patient():
    with pytest.raises(ValueError):
        Arm(pid=['a', 'a'], time=[1, 2], event=[0, 0], departed=[0, 1])


def test_scenario_events():
    a, _ = toy()
    rate = stayer_rate(a)
    sc = {s.name: s for s in build_scenarios((.25, .5, .75), rate, False)}
    draws = generate_draws({'evaluation': a}, (.25, .5, .75), rate, 5, 7, False)
    A = Assignments({'evaluation': a}, draws, 5)
    assert np.array_equal(A.events(sc['as_analyzed']), a.event)
    ev_all = A.events(sc['all_departed_died'])
    assert ev_all[a.departed].all() and np.array_equal(ev_all[~a.departed], a.event[~a.departed])
    n_dep = a.departed_ids.size
    for d in range(5):
        assert len(A.dead_ids(sc['fraction_0.50'], d, 'evaluation')) == round(.5 * n_dep)
        assert len(A.dead_ids(sc['stayer_rate'], d, 'evaluation')) == round(rate * n_dep)


def test_route_informed_respects_codes():
    a, _ = toy(route=True)
    rate = stayer_rate(a)
    sc = {s.name: s for s in build_scenarios((.25,), rate, True)}
    A = Assignments({'evaluation': a}, generate_draws({'evaluation': a}, (.25,), rate, 20, 3, True), 20)
    code = a.route_of()
    for d in range(20):
        dead = set(A.dead_ids(sc['route_informed'], d, 'evaluation'))
        assert all(code[p] != 'censored' for p in dead)                       # remain censored
        assert {p for p in a.departed_ids if code[p] == 'dead'} <= dead       # always dead


def test_between_patient_c_equals_sksurv_with_one_row_per_patient():
    rng = np.random.default_rng(5)
    n = 400
    time = rng.integers(1, 20, size=n).astype(float)                         # ties in time
    event = rng.random(n) < .5
    risk = np.round(rng.random(n), 1)                                          # ties in risk
    pid = np.array([f'x{i}' for i in range(n)])
    got = between_patient_c(event, time, pid, {'m': risk}, chunk=64)['m']
    assert got == pytest.approx(harrell_c(event, time, risk), abs=1e-12)


def test_oe_crossing():
    c = oe_crossing([(0, .9), (.25, .98), (.5, 1.06), (1, 1.2)])
    assert c['interpolated_crossing'] == pytest.approx(.25 + .02 * .25 / .08)
    assert oe_crossing([(0, .9), (1, .95)]) is None


def test_run_end_to_end_small():
    a, rng = toy(n_pt=150)
    true = rng.random(a.pid.size)
    risk = {'ref': true + rng.normal(0, .3, a.pid.size), 'alt': true + rng.normal(0, .5, a.pid.size)}
    risk_h = {m: 1 / (1 + np.exp(-v)) for m, v in risk.items()}
    res = run(a, Config(risk=risk, reference='ref', risk_h=risk_h, horizon=3 * 365, n_draws=10, n_boot=30,
                        between_patient=True), log=lambda *_: None)
    d = res.discrimination.set_index('scenario')
    for r in res.paired.itertuples():
        assert r.delta == pytest.approx(d.loc[r.scenario, 'c_alt'] - d.loc[r.scenario, 'c_ref'])
    assert set(res.paired.scenario) == {'as_analyzed', 'stayer_rate', 'all_departed_died'}
    assert res.calibration is not None and res.calibration.n_km.nunique() == 1   # O:E denominator fixed


# ---------------------------------------------------------------- regressions from the Codex review (2026-09-30)
def test_flags_are_parsed_strictly():
    a = Arm(pid=['a', 'b', 'c'], time=[1, 2, 3], event=['0', '1', '0'], departed=['no', 'no', 'yes'])
    assert a.event.tolist() == [False, True, False] and a.departed.tolist() == [False, False, True]
    for bad in (['0', '2', '0'], [0, np.nan, 0], ['maybe', '0', '0']):
        with pytest.raises(ValueError):
            Arm(pid=['a', 'b', 'c'], time=[1, 2, 3], event=bad, departed=[0, 0, 0])
    with pytest.raises(ValueError):
        Arm(pid=['a', None, 'c'], time=[1, 2, 3], event=[0, 1, 0], departed=[0, 0, 0])


def test_cli_keeps_identifiers_as_written(tmp_path):
    from types import SimpleNamespace
    from censoring_impact.cli import _arm, _read
    f = tmp_path / 'd.csv'
    f.write_text('patient_id,time,event,departed,risk\n001,1,1,0,2\n1,4,0,0,1\nNA,2,0,1,3\n3,3,0,1,0\n')
    a = SimpleNamespace(id='patient_id', time='time', event='event', departed='departed', route=None)
    arm = _arm(_read(str(f), a), a, 'test')
    assert arm.pid.tolist() == ['001', '1', 'NA', '3'] and stayer_rate(arm) == 0.5


def test_fraction_names_are_distinct_and_validated():
    assert [fraction_name(f) for f in (.25, .5, .125, .251, .254)] == [
        'fraction_0.25', 'fraction_0.50', 'fraction_0.125', 'fraction_0.251', 'fraction_0.254']
    for bad in ((.25, .25), (.25001,), (0,), (1,)):
        with pytest.raises(ValueError):
            build_scenarios(bad, .5, False)


def test_assigned_counts_round_halves_to_even():
    a = Arm(pid=[f'p{i}' for i in range(12)], time=[1] * 12, event=[1, 0] + [0] * 10, departed=[0, 0] + [1] * 10)
    draws = generate_draws({'evaluation': a}, (.25, .75), .5, 3, 1, False)
    k = {sc: set(g['mask'].map(lambda m: m.count('1'))) for sc, g in draws.groupby('scenario')}
    assert k == {'fraction_0.25': {2}, 'fraction_0.75': {8}, 'stayer_rate': {5}}      # 2.5 -> 2, 7.5 -> 8


def test_oe_kept_when_the_slope_cannot_be_fitted():
    e, t = np.array([1, 0, 0], bool), np.array([2., 3., 4.])
    r = calibration(e, t, np.arange(3), {'m': np.array([.2, .3, .4])}, 1.0, censoring_estimator(e, t))['m']
    assert r['oe'] == 0 and r['expected'] == pytest.approx(.3) and np.isnan(r['slope']) and r['n_slope'] == 0


def test_constant_predicted_risk_gives_no_slope():
    e, t = np.array([1, 0, 0, 0], bool), np.array([1., 3., 3., 3.])
    r = calibration(e, t, np.arange(4), {'m': np.full(4, .5)}, 2.0, censoring_estimator(e, t))['m']
    assert np.isnan(r['slope']) and np.isfinite(r['oe'])


def test_invalid_inputs_are_rejected():
    a, rng = toy(n_pt=60)
    risk = {'m': rng.random(a.pid.size)}
    with pytest.raises(ValueError):                                   # probabilities outside 0-1
        run(a, Config(risk=risk, reference='m', risk_h={'m': risk['m'] + 1}, horizon=365, n_draws=2, n_boot=0), log=lambda *_: None)
    with pytest.raises(ValueError):                                   # everyone departed
        run(Arm(pid=['a', 'b'], time=[1, 2], event=[0, 0], departed=[1, 1]), Config(risk={'m': [1., 2.]}, reference='m'), log=lambda *_: None)
    with pytest.raises(ValueError):                                   # no deaths at all
        run(Arm(pid=['a', 'b', 'c'], time=[1, 2, 3], event=[0, 0, 0], departed=[0, 0, 1]),
            Config(risk={'m': [1., 2., 3.]}, reference='m'), log=lambda *_: None)


def test_oe_crossing_exact_and_first_sign_change():
    assert oe_crossing([(0, .8), (.4, 1.0), (1, 1.2)])['interpolated_crossing'] == .4
    assert oe_crossing([(0, .9), (.5, 1.1), (.75, .95), (1, 1.2)])['interpolated_crossing'] == pytest.approx(.25)


def test_single_model_report_is_written(tmp_path):
    from censoring_impact.report import write
    a, rng = toy(n_pt=80)
    cfg = Config(risk={'only': rng.random(a.pid.size)}, reference='only', n_draws=3, n_boot=10)
    res = run(a, cfg, log=lambda *_: None)
    files = write(res, cfg, tmp_path, {})
    assert (tmp_path / 'summary.md').exists() and res.paired.empty and len(files) >= 5
