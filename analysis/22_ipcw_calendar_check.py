# -*- coding: utf-8 -*-
"""
校準權重的設限分布與評估族群是否一致(2026-09-29,回應 Codex 第四輪)。

09 的 IPCW 設限分布由「全部」發展組 landmark 估計,但斜率只在「3 年 horizon 可完整觀察」的測試 landmark
(landmark 年 ≤ 截止年)上評估。較晚年份的發展組 landmark 會因資料截止而在 3 年前設限,可能讓權重偏離評估族群。
這裡比較三種做法下的 3 年校準斜率(模型固定、同一組指派):
  A 原做法:全部發展組估計設限分布
  B 限定:只用 landmark 年 ≤ 截止年的發展組估計設限分布
  C 不加權:只在極端情境有意義(所有可評估 landmark 的 3 年狀態都已知)
重用 09 的設定段,不跑 bootstrap、不改寫 S4。輸出 outputs/tables/S9_ipcw_calendar_check.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator

HERE = Path(__file__).parent
src = (HERE/'09_calibration_sensitivity.py').read_text(encoding='utf-8')
ns = {'__name__': 'ipcw_check', '__file__': str(HERE/'09_calibration_sensitivity.py')}
exec(compile(src[:src.index('# ---- 實作驗證')], '09_setup', 'exec'), ns)
S, calib, cde_for, SUB_IDX, H = ns['S'], ns['calib'], ns['cde_for'], ns['SUB_IDX'], ns['H']

test_years = S.test.landmark_year.to_numpy()
cut = int(test_years[S.elig3].max())
assert (test_years[~S.elig3] > cut).all(), '測試組的日曆合格條件不是單一截止年'
dev_years = S.dev.landmark_year.to_numpy()
DEV_OK = dev_years <= cut

def cde_calendar(scenario, draw=0):
    ev = S.events(scenario, draw, arm='dev')
    return CensoringDistributionEstimator().fit(Surv.from_arrays(ev[DEV_OK], S.dev_tm[DEV_OK]))

def unweighted(ev, idx):
    t, e = S.tm[idx], ev[idx]
    known = (e & (t <= H)) | (t > H)
    y = (e & (t <= H))[known].astype(int)
    out = {}
    for m in S.MODELS:
        r = S.risk3[m][idx][known]
        x = np.log(np.clip(r, 1e-6, 1-1e-6) / np.clip(1-r, 1e-6, 1))
        out[m] = float(LogisticRegression(penalty=None, max_iter=1000).fit(x[:, None], y).coef_[0, 0])
    return out, int(known.sum())

rows = []
for sc in ('primary', 'matched_rate', 'worst'):
    nd = S.n_draws(sc)
    acc = {(m, k): [] for m in S.MODELS for k in ('A', 'B', 'C')}
    nk = []
    for d in range(nd):
        ev = S.events(sc, d)
        ra, rb = calib(ev, SUB_IDX, cde_for(sc, d)), calib(ev, SUB_IDX, cde_calendar(sc, d))
        rc, n_known = unweighted(ev, SUB_IDX); nk.append(n_known)
        for m in S.MODELS:
            acc[(m, 'A')].append(ra[m]['slope']); acc[(m, 'B')].append(rb[m]['slope']); acc[(m, 'C')].append(rc[m])
    for m in S.MODELS:
        rows.append(dict(scenario=sc, model=m, calendar_cutoff=cut, dev_landmarks_all=len(dev_years),
                         dev_landmarks_calendar=int(DEV_OK.sum()), n_known_mean=float(np.mean(nk)),
                         slope_all_dev=float(np.mean(acc[(m, 'A')])), slope_calendar_dev=float(np.mean(acc[(m, 'B')])),
                         slope_unweighted=float(np.mean(acc[(m, 'C')]))))
F = pd.DataFrame(rows)
S4 = pd.read_csv(S.TAB/'S4_calibration_sensitivity.csv').set_index(['scenario', 'model'])
for r in F.itertuples():                                    # 原做法必須重現 S4
    assert abs(r.slope_all_dev - S4.loc[(r.scenario, r.model), 'slope']) < 1e-9, (r.scenario, r.model)
F['diff_calendar_minus_all'] = F.slope_calendar_dev - F.slope_all_dev
F.to_csv(S.TAB/'S9_ipcw_calendar_check.csv', index=False)
pd.set_option('display.width', 200)
print(f'截止年 {cut};發展組 landmark 全部 {len(dev_years)}、限定 {int(DEV_OK.sum())}')
print(F.drop(columns=['calendar_cutoff', 'dev_landmarks_all', 'dev_landmarks_calendar']).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f'\n限定發展組後斜率的最大變動 {F.diff_calendar_minus_all.abs().max():.4f}')
w = F[F.scenario == 'worst']
print(f'極端情境加權與不加權的最大差 {(w.slope_all_dev - w.slope_unweighted).abs().max():.4f}(n_known {w.n_known_mean.iloc[0]:.0f})')
print(f'寫出 {S.TAB}/S9_ipcw_calendar_check.csv')
