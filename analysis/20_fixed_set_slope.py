# -*- coding: utf-8 -*-
"""
3 年校準斜率的變動拆解(2026-09-28,回應 Codex 交叉審查)。

斜率只能用 3 年狀態已知的 landmark。重新指派離院者會 (a) 讓原本 3 年前就設限的 landmark 變成可評估,
(b) 依情境重估 IPCW 權重。這裡在「每個情境都可評估的固定集合」上重算斜率:固定集合的 3 年狀態不變,
所以它的變動只來自 (b);與全部可評估集合的斜率相比,就知道 (a) 占多少。

重用 09 的設定段(資料、權重、calib 函式),只執行到點估計之前,不跑 bootstrap、不改寫 S4。
輸出 outputs/tables/S8_fixed_set_slope.csv
"""
import os, sys
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE/'09_calibration_sensitivity.py').read_text(encoding='utf-8')
ns = {'__name__': 'fixed_set_slope', '__file__': str(HERE/'09_calibration_sensitivity.py')}
exec(compile(src[:src.index('# ---- 各情境的點估計')], '09_setup', 'exec'), ns)
S, calib, cde_for, FIXED, SUB_IDX = ns['S'], ns['calib'], ns['cde_for'], ns['FIXED'], ns['SUB_IDX']

rows = []
for sc in ('primary', 'matched_rate', 'worst'):
    nd = S.n_draws(sc)
    for m in S.MODELS:
        fx = [calib(S.events(sc, d), FIXED, cde_for(sc, d))[m]['slope'] for d in range(nd)]
        fu = [calib(S.events(sc, d), SUB_IDX, cde_for(sc, d))[m]['slope'] for d in range(nd)]
        rows.append(dict(scenario=sc, model=m, n_fixed=len(FIXED), slope_fixed_set=float(np.mean(fx)),
                         slope_all_evaluable=float(np.mean(fu))))
F = pd.DataFrame(rows)
ref = F[F.model == S.REF].set_index('scenario')
assert abs(ref.loc['primary', 'slope_fixed_set'] - ref.loc['primary', 'slope_all_evaluable']) < 1e-12
F.to_csv(S.TAB/'S8_fixed_set_slope.csv', index=False)
print(F.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
w = ref.loc['worst', 'slope_fixed_set'] - ref.loc['primary', 'slope_fixed_set']
t = ref.loc['worst', 'slope_all_evaluable'] - ref.loc['primary', 'slope_all_evaluable']
print(f'\n參照模型 primary→worst:全部可評估集合 {t:+.3f};其中權重重估(固定集合){w:+.3f}')
print(f'寫出 {S.TAB}/S8_fixed_set_slope.csv')
