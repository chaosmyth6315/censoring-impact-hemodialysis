# -*- coding: utf-8 -*-
"""
設限敏感度的配對 bootstrap。

比照 dynamic_v1 的協定:以病人為叢集重抽樣,重建其全部 landmark,在同一組重抽樣上
算每個模型的 C,再取配對差異。模型不重配適。

**點估計直接讀 S1**,不在此重算,以保證 Table S2 的絕對 C 相減會等於 Table 3 的 ΔC。
(交叉審查抓到舊版兩處各自抽樣,相減對不起來。)

隨機情境的指派在每一次 replicate 內從共用抽樣池重抽,因此區間同時涵蓋病人抽樣與
「哪些離院者其實死了」的指定不確定性;病人重抽樣與指派使用獨立亂數流。

**區間涵蓋什麼、不涵蓋什麼**,寫進 S2 的欄位並在正文載明:
涵蓋 = 固定切分下對 292 位測試病人的重抽樣 + 指派身分。
不涵蓋 = 模型配適與超參數選擇、插補、切分本身、留院者死亡比例的估計誤差、
死亡發生的時點、以及設限敏感度模型本身的正確性。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _scenarios as S
from sksurv.metrics import concordance_index_censored

TAB = S.TAB; N_BOOT = 1000
SCEN = ['primary', 'matched_rate', 'route_informed', 'worst']
S1 = pd.read_csv(TAB/'S1_censoring_sensitivity.csv').set_index('scenario')

def cw(idx, ev):
    e, t = ev[idx], S.tm[idx]
    if e.sum() < 20: return None
    try:
        return {m: float(concordance_index_censored(e, t, S.risk[m][idx])[0]) for m in S.MODELS}
    except Exception:
        return None

rng_pt  = np.random.default_rng(S.SEED)        # 只用於病人重抽樣
rng_dr  = np.random.default_rng(S.SEED + 1)    # 只用於挑選指派抽樣
draws = {s: [] for s in SCEN}
t0 = time.time()
for b in range(N_BOOT):
    pick = rng_pt.choice(S.patients, size=len(S.patients), replace=True)
    idx = np.concatenate([S.groups[p] for p in pick])
    for s in SCEN:
        d = 0 if s in S.DETERMINISTIC else int(rng_dr.integers(S.N_DRAWS))
        r = cw(idx, S.events(s, d))
        if r is not None: draws[s].append(r)
    if (b+1) % 250 == 0: print(f'  {b+1}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)

rows = []
print('\n' + '='*104)
print('ΔC 相對 current-value Cox(病人叢集配對 bootstrap,95% 百分位區間)')
print('點估計取自 S1,與 Table S2 的絕對 C 完全一致')
print('='*104)
for s in SCEN:
    print(f'\n[{s}]  有效 replicate {len(draws[s])}/{N_BOOT}'
          + ('   (區間含指派不確定性)' if s not in S.DETERMINISTIC else ''))
    for m in S.MODELS:
        if m == S.REF: continue
        d = np.array([x[m] - x[S.REF] for x in draws[s]])
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = float(S1.loc[s, f'c_{m}'] - S1.loc[s, f'c_{S.REF}'])
        sd = (np.std([S.events(s, k) for k in range(0)], ddof=1) if False else np.nan)
        rows.append(dict(scenario=s, model=m, reference=S.REF,
                         c_model=float(S1.loc[s, f'c_{m}']), c_reference=float(S1.loc[s, f'c_{S.REF}']),
                         delta=pt, ci_low=lo, ci_high=hi,
                         excludes_zero=bool(lo > 0 or hi < 0), n_boot=len(d)))
        print(f'  {S.NICE[m]:26s} {pt:+.4f}  ({lo:+.4f}, {hi:+.4f})' + (' *' if lo > 0 or hi < 0 else ''))

# 指派不確定性單獨量化(不重抽病人)
ALL = np.arange(len(S.pid))
for s in SCEN:
    if s in S.DETERMINISTIC: continue
    for m in S.MODELS:
        if m == S.REF: continue
        v = [cw(ALL, S.events(s, d)) for d in range(S.N_DRAWS)]
        sd = float(np.std([x[m] - x[S.REF] for x in v], ddof=1))
        for r in rows:
            if r['scenario'] == s and r['model'] == m: r['assignment_sd'] = sd
for r in rows: r.setdefault('assignment_sd', np.nan)

B = pd.DataFrame(rows)
B['covers'] = 'test-patient resampling; assignment identity'
B['excludes'] = 'model fitting; hyperparameter selection; imputation; the split; the stayer mortality estimate; timing of death after departure'
B.to_csv(TAB/'S2_sensitivity_bootstrap.csv', index=False)

# 一致性斷言:點估計必須等於 S1 相減
for r in rows:
    assert abs(r['delta'] - (S1.loc[r['scenario'], f'c_{r["model"]}']
                             - S1.loc[r['scenario'], f'c_{S.REF}'])) < 1e-12
for m in ('random_survival_forest', 'gradient_boosted_survival'):
    g = B[B.model == m]
    assert not ((g.delta > 0) & g.excludes_zero).any(), f'{m} 在某情境下顯著勝過 Cox'

print('\n' + '-'*104); print('排序結論在各情境下是否一致'); print('-'*104)
def verdict(r):
    return ('顯著較佳' if r.excludes_zero and r.delta > 0 else
            '顯著較差' if r.excludes_zero and r.delta < 0 else '無法區分')
for m in S.MODELS:
    if m == S.REF: continue
    vs = [verdict(B[(B.scenario == s) & (B.model == m)].iloc[0]) for s in SCEN]
    print(f'  {S.NICE[m]:26s} ' + ' → '.join(f'{s} {v}' for s, v in zip(SCEN, vs))
          + f'   {"一致" if len(set(vs))==1 else "!! 改變"}')
print(f'\n寫出 {TAB}/S2_sensitivity_bootstrap.csv  用時 {time.time()-t0:.0f}s')
