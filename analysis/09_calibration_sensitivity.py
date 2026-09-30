# -*- coding: utf-8 -*-
"""
設限對 3 年校準的影響 —— 與判別力同一套作法:模型不重配適,預測機率鎖定,
只改結局定義。

交叉審查後的三項修正:

1. **權重依情境重估。** 情境是對世界的假設,不是只對測試集的操作,所以同一個指派也
   套用到發展組,再由該情境下的發展組估計設限分布。舊版三個情境共用原始的 G0(t),
   會讓被指派為死亡的人拿到本不該有的權重。
2. **補上病人叢集 bootstrap。** 校準是本文的頭條發現,舊版卻只有點估計。
3. **掃描完整比例**,以定出 O:E 由 <1 轉為 >1 的跨越點,而不是只報三個情境就宣稱
   「任何比例都會反轉」。

校準定義沿用 dynamic_v1:3 年二元結局(3 年內死亡 / 存活過 3 年 / 3 年前設限則排除),
IPCW 加權 logistic:logit(y) = a + b·logit(預測 3 年風險)。
另報 O:E = KM 觀察 3 年風險 ÷ 平均預測 3 年風險;O:E 用**全部合格 landmark**,
分母不隨情境改變,斜率則只能用 3 年狀態已知者。

先以 primary 重現鎖定的斜率作為實作驗證,不符即中止。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _scenarios as S
from sklearn.linear_model import LogisticRegression
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator

TAB = S.TAB; H = S.H3; N_BOOT = 1000
SUB = S.elig3                                   # 3 年 horizon 可觀察的 landmark
SUB_IDX = np.flatnonzero(SUB)
KEY = ['primary', 'matched_rate', 'route_informed', 'worst']

def cde_for(scenario, draw=0):
    """依情境重估設限分布:同一指派也套用到發展組。"""
    ev = S.events(scenario, draw, arm='dev')
    return CensoringDistributionEstimator().fit(Surv.from_arrays(ev, S.dev_tm))

def binary_and_weights(ev, idx, cde):
    t, e = S.tm[idx], ev[idx]
    b = np.full(t.size, -1, dtype=int)
    hit = e & (t <= H); free = t > H
    b[hit] = 1; b[free] = 0
    w = np.zeros(t.size)
    if hit.any():  w[hit]  = 1.0 / cde.predict_proba(t[hit])
    if free.any(): w[free] = 1.0 / cde.predict_proba(np.full(int(free.sum()), H))
    return b, w

def calib(ev, idx, cde, models=None):
    """回傳每個模型的 slope、O:E 與兩個分母。"""
    models = models or S.MODELS
    b, w = binary_and_weights(ev, idx, cde)
    ok = b >= 0
    if np.unique(b[ok]).size < 2 or not np.isfinite(w[ok]).all() or (w[ok] <= 0).any():
        return None
    kt, ks = kaplan_meier_estimator(ev[idx], S.tm[idx])
    observed = float(1 - (ks[kt <= H][-1] if np.any(kt <= H) else 1.0))
    out = {}
    for m in models:
        r = S.risk3[m][idx]
        x = np.log(np.clip(r[ok], 1e-6, 1-1e-6) / np.clip(1-r[ok], 1e-6, 1))
        f = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(
            x[:, None], b[ok], sample_weight=w[ok])
        expected = float(r.mean())
        out[m] = dict(slope=float(f.coef_[0, 0]), intercept=float(f.intercept_[0]),
                      observed=observed, expected=expected, oe=observed/expected,
                      n_km=len(idx), n_slope=int(ok.sum()))
    return out

# ---- 實作驗證:primary 必須重現鎖定值
LOCK = pd.read_csv(S.DV1/'05_results'/'tables'/'dynamic_survival_v1_calibration.csv')
l3 = LOCK[np.isclose(LOCK.horizon_days, H)].set_index('model')
cde0 = cde_for('primary')
ref = calib(S.events('primary'), SUB_IDX, cde0)
for m in S.MODELS:
    assert abs(ref[m]['slope'] - float(l3.loc[m, 'slope'])) < 5e-3, (m, ref[m]['slope'])
assert ref[S.MODELS[0]]['n_slope'] == int(l3.loc[S.MODELS[0], 'eligible_count'])
print(f'實作驗證通過:四個模型的 primary 斜率皆重現鎖定值(n_slope={ref[S.MODELS[0]]["n_slope"]})\n')

# ---- 固定集合檢查
FIXED_LOCAL = (S.events('primary')[SUB_IDX] & (S.tm[SUB_IDX] <= H)) | (S.tm[SUB_IDX] > H)
FIXED = SUB_IDX[FIXED_LOCAL]
_fx = {s: calib(S.events(s), FIXED, cde_for(s))[S.REF]['slope'] for s in ('primary', 'worst')}
print(f'固定集合 n={len(FIXED)},其中離院者 {int(S.dep[FIXED].sum())} 位,追蹤皆超過 3 年 = '
      f'{bool((S.tm[FIXED][S.dep[FIXED]] > H).all())}')
print(f'  固定集合斜率 primary {_fx["primary"]:.3f} / worst {_fx["worst"]:.3f}'
      f'  → 3 年狀態不變,差異僅來自權重重估\n')

# ---- 各情境的點估計
rows, t0 = [], time.time()
for sc in S.ALL_SCENARIOS:
    nd = S.n_draws(sc)
    acc = {m: {k: [] for k in ('slope','intercept','observed','expected','oe','n_km','n_slope')}
           for m in S.MODELS}
    for d in range(nd):
        r = calib(S.events(sc, d), SUB_IDX, cde_for(sc, d))
        if r is None: continue
        for m in S.MODELS:
            for k in acc[m]: acc[m][k].append(r[m][k])
    for m in S.MODELS:
        rec = {k: float(np.mean(v)) for k, v in acc[m].items()}
        rec['n_km'] = int(round(rec['n_km'])); rec['n_slope'] = int(round(rec['n_slope']))
        rows.append(dict(scenario=sc, label=S.LABEL[sc], model=m, n_draws=nd, **rec))
S4 = pd.DataFrame(rows)

# ---- bootstrap:病人叢集,四個關鍵情境
print('bootstrap(病人叢集,%d 次)…' % N_BOOT, flush=True)
rng_pt = np.random.default_rng(S.SEED); rng_dr = np.random.default_rng(S.SEED + 1)
bacc = {(s, m, q): [] for s in KEY for m in S.MODELS for q in ('slope', 'oe')}
CDE = {s: [cde_for(s, d) for d in range(S.n_draws(s))] for s in KEY}
for b in range(N_BOOT):
    pk = rng_pt.choice(S.patients, size=len(S.patients), replace=True)
    idx = np.concatenate([S.groups[p] for p in pk])
    idx = idx[S.elig3[idx]]
    if len(idx) < 200: continue
    for s in KEY:
        d = 0 if s in S.DETERMINISTIC else int(rng_dr.integers(S.N_DRAWS))
        r = calib(S.events(s, d), idx, CDE[s][d])
        if r is None: continue
        for m in S.MODELS:
            bacc[(s, m, 'slope')].append(r[m]['slope']); bacc[(s, m, 'oe')].append(r[m]['oe'])
    if (b+1) % 250 == 0: print(f'  {b+1}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)
for i, r in S4.iterrows():
    if r.scenario not in KEY: continue
    for q in ('slope', 'oe'):
        v = bacc[(r.scenario, r.model, q)]
        S4.loc[i, f'{q}_ci_low'], S4.loc[i, f'{q}_ci_high'] = np.percentile(v, [2.5, 97.5])
        S4.loc[i, f'{q}_n_boot'] = len(v)
# ---- 分箱校準(Figure 4A):分箱由預測風險十分位決定,情境之間固定,只有觀察風險會變
def calib_bins(ev, idx, n_groups=10):
    out = {}
    for m in S.MODELS:
        r = S.risk3[m][idx]
        cuts = np.unique(np.quantile(r, np.linspace(0, 1, n_groups + 1)))
        grp = np.clip(np.searchsorted(cuts[1:-1], r, side='right'), 0, cuts.size - 2)
        recs = []
        for gi in range(cuts.size - 1):
            k = grp == gi
            kt, ks = kaplan_meier_estimator(ev[idx][k], S.tm[idx][k])
            surv = ks[kt <= H][-1] if np.any(kt <= H) else 1.0
            recs.append(dict(group=gi + 1, n=int(k.sum()),
                             mean_predicted=float(r[k].mean()), observed=float(1 - surv)))
        out[m] = recs
    return out

bin_rows = []
_refbins = calib_bins(S.events('primary'), SUB_IDX)
for sc in KEY:
    nd = S.n_draws(sc)
    obs = {m: np.zeros(len(_refbins[m])) for m in S.MODELS}
    for d in range(nd):
        bi = calib_bins(S.events(sc, d), SUB_IDX)
        for m in S.MODELS: obs[m] += np.array([x['observed'] for x in bi[m]])
    for m in S.MODELS:
        for j, r in enumerate(_refbins[m]):
            bin_rows.append(dict(scenario=sc, model=m, group=r['group'], n=r['n'],
                                 mean_predicted=r['mean_predicted'],
                                 observed=float(obs[m][j] / nd)))
B4 = pd.DataFrame(bin_rows)
for m in S.MODELS:                       # 分箱邊界必須在情境間相同
    piv = B4[B4.model == m].pivot(index='group', columns='scenario', values='mean_predicted')
    assert np.allclose(piv.values, piv.values[:, [0]]), m
B4.to_csv(TAB/'S4_calibration_bins.csv', index=False)

S4['reference_slope_fixed_primary'] = _fx['primary']; S4['n_fixed_set'] = len(FIXED)   # 參考模型的值(各模型自己的見 S8)
assert S4.n_km.nunique() == 1, 'O:E 的分母不應隨情境改變'
S4.to_csv(TAB/'S4_calibration_sensitivity.csv', index=False)

g = S4[S4.model == S.REF].set_index('scenario')
print('\n' + '='*104); print('3 年校準隨設限假設的變化(current-value Cox)'); print('='*104)
print(f'{"情境":30s} {"O:E":>8s} {"95% CI":>22s} {"slope":>8s} {"95% CI":>22s} {"n_slope":>8s}')
for sc in S.ALL_SCENARIOS:
    r = g.loc[sc]
    ci = (f'({r.oe_ci_low:.3f}, {r.oe_ci_high:.3f})' if sc in KEY else '—')
    cs = (f'({r.slope_ci_low:.3f}, {r.slope_ci_high:.3f})' if sc in KEY else '—')
    print(f'  {S.LABEL[sc]:28s} {r.oe:8.3f} {ci:>22s} {r.slope:8.3f} {cs:>22s} {int(r.n_slope):8d}')

# O:E 的跨越點
frac = {'primary': 0.0, 'fraction_0.25': .25, 'fraction_0.50': .50,
        'matched_rate': S.RATE_PATIENT, 'fraction_0.75': .75, 'worst': 1.0}
pts = sorted(((frac[s], float(g.loc[s].oe)) for s in frac), key=lambda x: x[0])
below = [p for p in pts if p[1] < 1][-1]; above = [p for p in pts if p[1] > 1][0]
cross = below[0] + (1-below[1])*(above[0]-below[0])/(above[1]-below[1])
print(f'\nO:E 由 <1 轉為 >1 的跨越點:介於 {100*below[0]:.0f}% (O:E {below[1]:.3f}) '
      f'與 {100*above[0]:.0f}% (O:E {above[1]:.3f}) 之間,線性內插約 {100*cross:.0f}%')
print('  → 不可宣稱「任何比例的離院者死亡都會反轉方向」')
S4.attrs = {}
pd.DataFrame([dict(crossing_lower_fraction=below[0], crossing_lower_oe=below[1],
                   crossing_upper_fraction=above[0], crossing_upper_oe=above[1],
                   interpolated_crossing=cross)]).to_csv(TAB/'S4_oe_crossing.csv', index=False)
print(f'\n寫出 {TAB}/S4_calibration_sensitivity.csv 與 S4_oe_crossing.csv  用時 {time.time()-t0:.0f}s')
