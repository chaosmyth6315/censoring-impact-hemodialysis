# -*- coding: utf-8 -*-
"""
離院者與留院者的預測 3 年風險(2026-09-28 新增,聚焦版 Paper C 的小分析)。

審稿人一定會問「轉出去的是比較病還是比較健康的人」。模型已鎖定,這裡只讀鎖定的
預測值,比較測試集中離院病人與留院病人 landmark 的預測 3 年風險。不重配適、不改結局。

- landmark 層級:四個模型各自的平均、中位數(IQR)。
- 病人層級:每位病人先取自己各 landmark 的平均,避免 landmark 多的人權重較大。
- 差值(離院 − 留院)的病人叢集 bootstrap 區間,1,000 次,依病人重抽。

輸出 outputs/tables/S6_departure_predicted_risk.csv
"""
import os
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

DV1 = Path(os.environ.get('LOCKED_DIR', 'locked_model_outputs'))
TAB = Path(os.environ.get('RESULTS_DIR', 'results'))
MODELS = ['current_value_cox', 'trajectory_cox', 'random_survival_forest', 'gradient_boosted_survival']
SEED, B = 20260928, 1000

PR = pd.read_csv(DV1/'05_results'/'tables'/'dynamic_survival_v1_predictions.csv')
OC = pd.read_csv(DV1/'02_longitudinal_data'/'outcomes.csv')
DEPARTED = set(OC.loc[OC['是否離開本院'] == 1, 'analysis_id'])

ref = PR[PR.model == MODELS[0]].reset_index(drop=True)
pid = ref.analysis_id.to_numpy()
dep = ref.analysis_id.isin(DEPARTED).to_numpy()
assert len(ref) == 1355 and dep.sum() == 293 and len(set(pid[dep])) == 72
patients = np.unique(pid)
rows_of = {p: np.flatnonzero(pid == p) for p in patients}
dep_pt = np.array([p in DEPARTED for p in patients])

rng = np.random.default_rng(SEED)
boot_pt = [rng.choice(patients, size=len(patients), replace=True) for _ in range(B)]

out = []
for m in MODELS:
    g = PR[PR.model == m].reset_index(drop=True)
    assert (g.analysis_id.to_numpy() == pid).all()
    assert (g.landmark_year.to_numpy() == ref.landmark_year.to_numpy()).all()
    r = g['risk_1095.75'].to_numpy(float)
    pt_mean = np.array([r[rows_of[p]].mean() for p in patients])

    def diff(idx_pt):
        rows = np.concatenate([rows_of[p] for p in idx_pt])
        d = dep[rows]
        return r[rows][d].mean() - r[rows][~d].mean()

    est = r[dep].mean() - r[~dep].mean()
    bs = np.array([diff(b) for b in boot_pt])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    q = lambda x: np.percentile(x, [25, 50, 75])
    out.append(dict(
        model=m,
        n_lm_departed=int(dep.sum()), n_lm_remained=int((~dep).sum()),
        n_pt_departed=int(dep_pt.sum()), n_pt_remained=int((~dep_pt).sum()),
        mean_departed=r[dep].mean(), mean_remained=r[~dep].mean(),
        q25_departed=q(r[dep])[0], median_departed=q(r[dep])[1], q75_departed=q(r[dep])[2],
        q25_remained=q(r[~dep])[0], median_remained=q(r[~dep])[1], q75_remained=q(r[~dep])[2],
        diff_mean=est, diff_ci_low=lo, diff_ci_high=hi, n_boot=B,
        pt_mean_departed=pt_mean[dep_pt].mean(), pt_mean_remained=pt_mean[~dep_pt].mean(),
        pt_diff_mean=pt_mean[dep_pt].mean() - pt_mean[~dep_pt].mean()))

S6 = pd.DataFrame(out)
S6.to_csv(TAB/'S6_departure_predicted_risk.csv', index=False)
print('預測 3 年風險:離院病人 vs 留院病人(測試集,模型鎖定)')
for _, x in S6.iterrows():
    print(f'  {x.model:26s} 離院 {x.mean_departed:.3f} [中位 {x.median_departed:.3f} ({x.q25_departed:.3f}–{x.q75_departed:.3f})]'
          f'  留院 {x.mean_remained:.3f} [中位 {x.median_remained:.3f} ({x.q25_remained:.3f}–{x.q75_remained:.3f})]'
          f'  差 {x.diff_mean:+.3f} ({x.diff_ci_low:+.3f}, {x.diff_ci_high:+.3f})'
          f'  病人層級差 {x.pt_diff_mean:+.3f}')
print(f'\n寫出 {TAB}/S6_departure_predicted_risk.csv')
