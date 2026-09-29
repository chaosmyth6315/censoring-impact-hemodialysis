# -*- coding: utf-8 -*-
"""
設限敏感度的判別力部分 —— 第三篇的關鍵前提。

世代裡 1,810 人中有 556 人離開本院(465 人轉院),離院後的死亡看不到:全體只有
2 位離院者被記錄為死亡。測試集 293 個 landmark(21.6%)屬於離院病人,其中事件數為 0。
標準存活分析假設非資訊性設限,在此明確被違反,方向是低估死亡。

作法:模型不重新配適。dynamic_v1 的風險分數是鎖定的,只改結局定義,
因此測到的是**排序表現對設限假設的穩健性**,而不是重新訓練後的表現。

情境與指派抽樣一律取自 `_scenarios.py`,與 02、09 共用同一組抽樣(交叉審查曾抓到
本檔與 02 各自抽樣,導致絕對 C 與 ΔC 相減對不起來)。

另附「排除同病人配對」的敏感度:鎖定的全域 Harrell C 把同一病人不同年度的 landmark
當成可互相比較的配對。這裡重算一個只用跨病人配對的 C,用**同一組 200 次抽樣**,
所以差異純粹來自配對定義。它不是 patient-level 的估計量,只量化這一項的影響。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _scenarios as S
from sksurv.metrics import concordance_index_censored

TAB = S.TAB

_SAME = S.pid[:, None] == S.pid[None, :]

def c_index(ev, risk):
    return float(concordance_index_censored(ev, S.tm, risk)[0])

def c_between(ev):
    """只用跨病人配對的 Harrell C,四個模型一次算完。

    可比較配對:較早者發生事件,或同一時間點的事件對設限(與 sksurv 相同)。並列風險以 sksurv 0.28 的 1e-8 容忍度計 0.5,
    與 `concordance_index_censored` 一致。可比較遮罩與模型無關,只算一次。
    """
    # 與 sksurv 相同的可比較配對:較早者為事件;同一時間點時,事件與「設限」者也可比較(Codex 2026-09-29 指出原本漏了這一條)
    comparable = (ev[:, None] & ((S.tm[:, None] < S.tm[None, :]) |
                                 ((S.tm[:, None] == S.tm[None, :]) & ~ev[None, :]))) & ~_SAME
    n = int(comparable.sum())
    if not n: return {m: float('nan') for m in S.MODELS}
    out = {}
    for m in S.MODELS:
        r = S.risk[m][:, None]; d = r - r.T
        conc = (d > 1e-8) & comparable
        ties = (np.abs(d) <= 1e-8) & comparable
        out[m] = float((int(conc.sum()) + .5*int(ties.sum())) / n)
    return out

rows = []
print('=' * 112)
print(f'判別力:模型不重配適,只改結局定義。隨機情境取 {S.N_DRAWS} 次指派的平均。')
print('=' * 112)
for sc in S.ALL_SCENARIOS:
    nd = S.n_draws(sc)
    acc = {m: [] for m in S.MODELS}; acc_x = {m: [] for m in S.MODELS}
    nev, ndead = [], []
    for d in range(nd):
        ev = S.events(sc, d)
        for m in S.MODELS: acc[m].append(c_index(ev, S.risk[m]))
        nev.append(int(ev.sum())); ndead.append(len(S.dead_patients(sc, d)))
    if sc in ('primary', 'matched_rate', 'worst'):     # 跨病人版本只算三個關鍵情境,但用相同的 200 次抽樣
        for d in range(nd):
            r_ = c_between(S.events(sc, d))
            for m in S.MODELS: acc_x[m].append(r_[m])
    c  = {m: float(np.mean(acc[m])) for m in S.MODELS}
    cx = {m: (float(np.mean(acc_x[m])) if acc_x[m] else np.nan) for m in S.MODELS}
    best = max(c, key=c.get)
    rows.append(dict(scenario=sc, label=S.LABEL[sc], n_draws=nd,
                     dead_assigned=float(np.mean(ndead)), n_events=float(np.mean(nev)),
                     event_rate=float(np.mean(nev))/len(S.pid), best_model=best,
                     **{f'c_{m}': c[m] for m in S.MODELS},
                     **{f'cx_{m}': cx[m] for m in S.MODELS}))
    print(f'  {S.LABEL[sc]:28s} 事件 {np.mean(nev):6.1f} ({100*np.mean(nev)/len(S.pid):4.1f}%)  ' +
          '  '.join(f'{m.split("_")[0][:5]} {c[m]:.4f}' for m in S.MODELS) +
          f'   最佳 {S.NICE[best]}', flush=True)

S1 = pd.DataFrame(rows); S1.to_csv(TAB/'S1_censoring_sensitivity.csv', index=False)
print('\n' + '-'*112); print('結論是否穩健'); print('-'*112)
print(f'  各情境的最佳模型: {[S.NICE[b] for b in S1.best_model.unique()]}')
assert S1.best_model.nunique() == 1, '最佳模型不應隨情境改變'
_g = lambda sc, m: float(S1[S1.scenario == sc][f'c_{m}'].iloc[0])
for m in S.MODELS:
    print(f'  {S.NICE[m]:26s} primary {_g("primary",m):.4f} → matched {_g("matched_rate",m):.4f}'
          f' → worst {_g("worst",m):.4f}   Δ {_g("worst",m)-_g("primary",m):+.4f}')
beat = [m for m in S.MODELS if m != S.REF and (S1[f'c_{m}'] > S1[f'c_{S.REF}']).all()]
print(f'\n  在所有情境下都勝過 current-value Cox 的模型: {[S.NICE[m] for m in beat] if beat else "無"}')
print('\n' + '-'*112); print('排除同病人配對後的 C（與主指標同一組 200 次抽樣）'); print('-'*112)
for sc in ('primary', 'matched_rate', 'worst'):
    r = S1[S1.scenario == sc].iloc[0]
    print(f'  {S.LABEL[sc]:28s} ' +
          '  '.join(f'{m.split("_")[0][:5]} {r[f"c_{m}"]:.4f}→{r[f"cx_{m}"]:.4f}' for m in S.MODELS))
_mx = max(abs(S1[S1.scenario == sc][f'c_{m}'].iloc[0] - S1[S1.scenario == sc][f'cx_{m}'].iloc[0])
          for sc in ('primary','matched_rate','worst') for m in S.MODELS)
print(f'\n  三個情境、四個模型的最大差 {_mx:.4f}')
for sc in ('primary','matched_rate','worst'):
    _r = S1[S1.scenario == sc].iloc[0]
    _oa = sorted(S.MODELS, key=lambda m: -_r[f'c_{m}']); _ox = sorted(S.MODELS, key=lambda m: -_r[f'cx_{m}'])
    print(f'  {sc:14s} 排序一致 = {_oa == _ox}')
    assert _oa == _ox, f'{sc}:排除同病人配對後排序改變,必須寫進正文'
print(f'\n寫出 {TAB}/S1_censoring_sensitivity.csv')
