"""
Checks that the web version (web/censoring_impact.html) computes what the Python tool computes.

Needs Node.js. The page's computing core is extracted and run in Node:
  1. Harrell's C, the censoring distribution and Kaplan-Meier on 40 random data sets with heavy ties, against
     scikit-survival: must be identical.
  2. The whole analysis on the synthetic example, against the Python tool with the same settings: in the scenarios
     without random assignment, C, between-patient C and O:E must be identical, and the calibration slope must equal a
     fully converged fit (the Python tool stops at scikit-learn's default tolerance, about 1e-4 away).
  3. Edge cases where the two once differed: rounding of assigned counts at .5, scenario names, a logistic fit that
     diverged, a constant predictor, the O:E crossing, the censoring distribution at a machine-epsilon boundary, O:E
     when the slope cannot be fitted, and malformed CSV files.
Run:  python tests/check_web.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.util import Surv

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from censoring_impact import Arm, Config, run  # noqa: E402
from censoring_impact.core import censoring_estimator  # noqa: E402

FAIL = []
TMP = Path(tempfile.mkdtemp())
page = (HERE/'web'/'censoring_impact.html').read_text(encoding='utf-8')
(TMP/'core.js').write_text(re.search(r'<script id="core">(.*?)</script>', page, re.S).group(1))


def check(what, ok, detail=''):
    print(f"  {'OK ' if ok else '!! '}{what}{'' if ok else '  -> ' + str(detail)}")
    if not ok:
        FAIL.append(what)


def node(script, *args):
    (TMP/'run.js').write_text(script)
    return json.loads(subprocess.run(['node', str(TMP/'run.js'), *args], capture_output=True, text=True, check=True).stdout)


print('=== 1. estimators against scikit-survival ===')
cases = []
for s in range(40):
    rng = np.random.default_rng(s); n = int(rng.integers(20, 400))
    t = np.round(rng.exponential(3, n), int(rng.integers(0, 3)))
    e = rng.random(n) < rng.uniform(.1, .9)
    if s == 5:
        e[:] = True
    e[0] = True
    r = np.round(rng.normal(size=n), int(rng.integers(1, 4)))
    if s % 4 == 0:
        r = r + rng.choice([0, 5e-9, 9e-9, 2e-8], n)          # near-ties around the 1e-8 tolerance
    cde = CensoringDistributionEstimator().fit(Surv.from_arrays(e, t))
    q = np.unique(np.r_[t[:30], np.quantile(t, [.1, .5, .9]), t.min() - 1])
    q = q[q < t.max()]
    kt, ks = kaplan_meier_estimator(e, t); H = float(np.quantile(t, .6))
    cases.append(dict(t=t.tolist(), e=e.astype(int).tolist(), r=r.tolist(), c=concordance_index_censored(e, t, r)[0],
                      q=q.tolist(), G=cde.predict_proba(q).tolist(), H=H,
                      km=float(ks[kt <= H][-1]) if (kt <= H).any() else 1.0))
(TMP/'cases.json').write_text(json.dumps(cases))
w = node("""const C = require('./core.js'), cases = require('./cases.json'); let w = { c: 0, G: 0, km: 0 };
for (const k of cases) { const t = Float64Array.from(k.t), e = Uint8Array.from(k.e), idx = Array.from(t, (_, i) => i);
  w.c = Math.max(w.c, Math.abs(C.harrellC(t, e, C.riskRanks(Float64Array.from(k.r)), idx) - k.c));
  const G = C.censoringEstimator(t, e); k.q.forEach((q, j) => w.G = Math.max(w.G, Math.abs(C.predictG(G, q) - k.G[j])));
  w.km = Math.max(w.km, Math.abs(C.kmAt(t, e, idx, k.H) - k.km)); }
console.log(JSON.stringify(w));""")
for k in ('c', 'G', 'km'):
    check(f'{k}: max |difference| {w[k]:.1e} over {len(cases)} data sets', w[k] == 0, w[k])

print('\n=== 2. whole analysis on the synthetic example ===')
csv = HERE/'examples'/'example_evaluation.csv'
if not csv.exists():                                     # a fresh download ships only the generator
    subprocess.run([sys.executable, str(HERE/'examples'/'make_example_data.py')], check=True, cwd=HERE)
d = pd.read_csv(csv)
arm = Arm(pid=d.patient_id.values, time=d.time.values, event=d.event.values, departed=d.departed.values,
          route=d.route.fillna('').values)
risk = {'Cox': d.risk_cox.values, 'Forest': d.risk_forest.values}
risk_h = {'Cox': d.risk3y_cox.values, 'Forest': d.risk3y_forest.values}
py = run(arm, Config(risk=risk, reference='Cox', risk_h=risk_h, horizon=3, calibration_eligible=d.horizon_observable.values,
                     n_draws=20, n_boot=50, between_patient=True), log=lambda *a: None)
js = node("""const fs = require('fs'), C = require('./core.js'), T = C.parseCSV(fs.readFileSync(process.argv[2], 'utf8'));
const col = n => T.rows.map(r => r[T.header.indexOf(n)]), num = n => Float64Array.from(col(n).map(Number));
const D = { pid: col('patient_id'), time: num('time'), event: Uint8Array.from(col('event').map(Number)),
  departed: Uint8Array.from(col('departed').map(Number)), route: col('route'), elig: Uint8Array.from(col('horizon_observable').map(Number)),
  risk: { Cox: num('risk_cox'), Forest: num('risk_forest') }, riskH: { Cox: num('risk3y_cox'), Forest: num('risk3y_forest') } };
C.runAnalysis(D, { fractions: [0.25, 0.5, 0.75], draws: 20, boot: 50, seed: 20260903, between: true, horizon: 3, reference: 'Cox' })
  .then(r => console.log(JSON.stringify(r)));""", str(csv))
dp, dj = py.discrimination.set_index('scenario'), pd.DataFrame(js['discrimination']).set_index('scenario')
cp = py.calibration.set_index(['scenario', 'model'])
cj = pd.DataFrame(js['calibration']).set_index(['scenario', 'model'])
check('stayer rate', js['exposure']['stayer_rate'] == py.exposure['stayer_rate'])
for sc in ('as_analyzed', 'all_departed_died'):
    dc = max(abs(dj.loc[sc, f'{p}{m}'] - dp.loc[sc, f'{p}{m}']) for p in ('c_', 'cx_') for m in risk)
    check(f'{sc}: C and between-patient C (max |diff| {dc:.1e})', dc == 0, dc)
    do = max(abs(cj.loc[(sc, m), q] - cp.loc[(sc, m), q]) for m in risk for q in ('oe', 'observed', 'expected'))
    check(f'{sc}: O:E, observed and expected risk (max |diff| {do:.1e})', do < 1e-12, do)
# fully converged slope in the analysis as conducted
t, e = d.time.values, d.event.values.astype(bool)
sub = np.flatnonzero(d.horizon_observable.values.astype(bool)); ts, es = t[sub], e[sub]
b = np.full(ts.size, -1); hit, free = es & (ts <= 3), ts > 3; b[hit], b[free] = 1, 0
cde = censoring_estimator(e, t); wt = np.zeros(ts.size)
wt[hit] = 1 / cde.predict_proba(ts[hit]); wt[free] = 1 / cde.predict_proba(np.full(free.sum(), 3.0))
ok = b >= 0
for m in risk:
    x = np.log(np.clip(risk_h[m][sub][ok], 1e-6, 1 - 1e-6) / np.clip(1 - risk_h[m][sub][ok], 1e-6, 1))[:, None]
    f = LogisticRegression(solver='newton-cg', C=np.inf, tol=1e-12, max_iter=100000).fit(x, b[ok], sample_weight=wt[ok])
    ds = abs(cj.loc[('as_analyzed', m), 'slope'] - f.coef_[0, 0])
    check(f'as_analyzed {m}: slope equals the converged fit (|diff| {ds:.1e}; Python tool differs by '
          f"{abs(cp.loc[('as_analyzed', m), 'slope'] - f.coef_[0, 0]):.1e})", ds < 1e-8, ds)

print('\n=== 3. edge cases where the two implementations once differed (Codex review, 2026-09-30) ===')
from censoring_impact.core import fraction_name, oe_crossing, calibration as py_cal  # noqa: E402
from censoring_impact.core import censoring_estimator as py_cde  # noqa: E402
vals = [.5, 1.5, 2.5, 3.5, 7.5, 2.4999, 18.0, 47.127]
fr = [.25, .5, .75, .125, .251, .254, .3333]
xs = np.repeat([-2., -2, 0, 0, 2, 2], [10, 3, 1, 9339, 3, 1573])        # once made the web solver diverge
ys = np.repeat([0, 1, 0, 1, 0, 1], [10, 3, 1, 9339, 3, 1573])
cases = dict(vals=vals, fr=fr, lx=xs.tolist(), ly=ys.tolist(),
             crossing=[[[0, .8], [.4, 1.0], [1, 1.2]], [[0, .9], [.5, 1.1], [.75, .95], [1, 1.2]], [[0, .9], [1, .95]]])
(TMP/'edge.json').write_text(json.dumps(cases))
J = node("""const C = require('./core.js'), k = require('./edge.json'), out = {};
out.round = k.vals.map(C.roundHalfEven); out.names = k.fr.map(C.fractionName);
out.fit = C.logisticFit(k.lx, k.ly, k.lx.map(() => 1)); out.flat = C.logisticFit([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]);
out.cross = k.crossing.map(C.oeCrossing);
const G = C.censoringEstimator(Float64Array.from([0.1, 1]), Uint8Array.from([0, 1])); out.g = C.predictG(G, 0.09999999999999999);
const cal = C.calibration(Uint8Array.from([1, 0, 0]), Float64Array.from([2, 3, 4]), [0, 1, 2], { m: Float64Array.from([.2, .3, .4]) }, 1,
  C.censoringEstimator(Float64Array.from([2, 3, 4]), Uint8Array.from([1, 0, 0]))); out.cal = cal.m;
out.csv = ['id,t\\n1,"2\\n', 'id,id\\n1,2\\n', 'id,t\\n1,2\\n3\\n'].map(t => { try { C.parseCSV(t); return 'accepted'; } catch (e) { return 'rejected'; } });
console.log(JSON.stringify(out, (key, v) => (typeof v === 'number' && !Number.isFinite(v)) ? null : v));""")
check('rounding of assigned counts: web equals Python round()', J['round'] == [round(v) for v in vals], J['round'])
check('scenario names: web equals Python', J['names'] == [fraction_name(f) for f in fr], J['names'])
ref = LogisticRegression(solver='newton-cg', C=np.inf, tol=1e-12, max_iter=100000).fit(xs[:, None], ys).coef_[0, 0]
check(f"logistic fit that once diverged now converges (web {J['fit']['slope']}, converged {ref:.8f})",
      J['fit']['slope'] is not None and abs(J['fit']['slope'] - ref) < 1e-6)
check('constant predictor gives no slope (NaN), not 0', J['flat']['slope'] is None)
py_cross = [None if (c := oe_crossing([tuple(p) for p in pts])) is None else c['interpolated_crossing'] for pts in cases['crossing']]
check('O:E crossing: web equals Python', all((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12)
                                            for a, b in zip(J['cross'], py_cross)), (J['cross'], py_cross))
e3, t3 = np.array([True, False]), np.array([0.1, 1.0])
g_ref = py_cde(~e3, t3).predict_proba(np.array([0.09999999999999999]))[0]
check(f"censoring distribution at a time within machine epsilon of an observation (web {J['g']}, sksurv {g_ref})", J['g'] == g_ref)
pc = py_cal(np.array([1, 0, 0], bool), np.array([2., 3., 4.]), np.arange(3), {'m': np.array([.2, .3, .4])}, 1.0,
            py_cde(np.array([1, 0, 0], bool), np.array([2., 3., 4.])))['m']
check('O:E kept when the slope cannot be fitted, same in both', J['cal']['oe'] == pc['oe'] == 0 and J['cal']['slope'] is None and np.isnan(pc['slope']))
check('malformed CSV is rejected (open quote, repeated header, short row)', J['csv'] == ['rejected'] * 3, J['csv'])
# assigned counts at .5 boundaries, whole analysis
ids = [f'p{i:02d}' for i in range(12)]
arm2 = Arm(pid=ids, time=np.arange(1, 13, dtype=float), event=[1, 0] + [0] * 10, departed=[0, 0] + [1] * 10)
py2 = run(arm2, Config(risk={'m': np.arange(12, dtype=float)}, reference='m', fractions=(.25, .75), n_draws=4, n_boot=0), log=lambda *_: None)
(TMP/'half.json').write_text(json.dumps(dict(pid=ids, time=list(range(1, 13)), event=[1, 0] + [0] * 10, departed=[0, 0] + [1] * 10)))
J2 = node("""const C = require('./core.js'), d = require('./half.json');
C.runAnalysis({ pid: d.pid, time: Float64Array.from(d.time), event: Uint8Array.from(d.event), departed: Uint8Array.from(d.departed),
  route: null, elig: null, risk: { m: Float64Array.from(d.time.map((_, i) => i)) }, riskH: null },
  { fractions: [.25, .75], draws: 4, boot: 0, seed: 1, between: false, horizon: 0, reference: 'm' })
  .then(r => console.log(JSON.stringify(Object.fromEntries(r.discrimination.map(x => [x.scenario, x.departed_assigned_death])))));""")
py_k = dict(zip(py2.discrimination.scenario, py2.discrimination.departed_assigned_death))
check(f'assigned deaths at .5 boundaries equal in both (web {J2}, Python {py_k})', J2 == py_k)

print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILED: ' + '; '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
