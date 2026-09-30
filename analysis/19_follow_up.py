# -*- coding: utf-8 -*-
"""
追蹤時間摘要(TRIPOD+AI 20a,2026-09-28 應作者要求補)。

追蹤時間 = 從 landmark 日期到死亡或設限的天數(鎖定檔的 target_duration_days),
依原分析的設限定義(離院即設限)。發展組與測試組各給中位數與四分位距;
另給反向 Kaplan–Meier 的中位潛在追蹤時間(把死亡當成設限),作為對照,不寫進正文。
輸出 outputs/tables/S7_follow_up.csv
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _scenarios as S
from sksurv.nonparametric import kaplan_meier_estimator

YR = 365.25
rows = []
for arm, tm, ev, pid in (('development', S.dev_tm, S.dev_ev0, S.dev_pid), ('test', S.tm, S.ev0, S.pid)):
    q25, q50, q75 = np.percentile(tm / YR, [25, 50, 75])
    t, s = kaplan_meier_estimator(ev, tm, reverse=True)         # 反向 KM(sksurv 規則:同一天的死亡先移出風險集)
    rkm = float(t[np.argmax(s <= 0.5)] / YR) if (s <= 0.5).any() else float('nan')
    rows.append(dict(arm=arm, landmarks=len(tm), patients=len(np.unique(pid)), deaths=int(ev.sum()),
                     median_years=q50, q25_years=q25, q75_years=q75, reverse_km_median_years=rkm))
F = pd.DataFrame(rows)
assert F.set_index('arm').loc['test', 'landmarks'] == 1355 and F.set_index('arm').loc['test', 'deaths'] == 656
assert F.set_index('arm').loc['development', 'landmarks'] == 5540
F.to_csv(S.TAB/'S7_follow_up.csv', index=False)
print(F.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
print(f'寫出 {S.TAB}/S7_follow_up.csv')
