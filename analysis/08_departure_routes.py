# -*- coding: utf-8 -*-
"""
離開途徑的細分,以及由它導出的 route-informed 設限情境。

matched_rate 把所有離院者一律以未離院者的事件率指定死亡。但離開途徑本身帶有
預後訊息:腎移植與腎功能恢復者的死亡率明顯較低,放棄透析者則近乎必死。
把途徑納入之後可以得到一個比 matched_rate 更貼近事實的中間情境。

指定規則（寫進 Methods,不可事後調整）:
  假定存活   腎移植、腎功能恢復        —— 這兩類的短期死亡率遠低於留院者
  假定死亡   放棄透析                  —— 停止透析後死亡在數週內
  依留院者事件率隨機指定  其餘全部（轉院、轉腹膜透析、未見於年報、不明）

輸出:S5 全體與測試集的途徑細分、route_informed 的病人層級指定規則。
"""
import os
import numpy as np, pandas as pd
from pathlib import Path

DV1 = Path(os.environ.get('LOCKED_DIR', 'locked_model_outputs'))
TAB = Path(os.environ.get('RESULTS_DIR', 'results'))

OC = pd.read_csv(DV1/'02_longitudinal_data'/'outcomes.csv')
PR = pd.read_csv(DV1/'05_results'/'tables'/'dynamic_survival_v1_predictions.csv')
test = PR[PR.model == 'current_value_cox']
test_ids = set(test.analysis_id)

# 英文標籤:年報的欄位是「中文 english」的合併字串,取英文段落
EN = {'轉院 transfer to another unit': 'Transfer to another dialysis unit',
      '未見於最終年報 absent from the final annual report': 'Absent from the final annual report',
      '腎移植 kidney transplantation': 'Kidney transplantation',
      '轉腹膜透析 change to peritoneal dialysis': 'Change to peritoneal dialysis',
      '狀態碼4(字典未列) status code 4': 'Status code 4 (not in the data dictionary)',
      '腎功能恢復 recovery of kidney function': 'Recovery of kidney function',
      '放棄 withdrawal': 'Withdrawal from dialysis',
      '不明原因退出 exit, reason not stated': 'Exit, reason not stated'}
ASSUME = {'腎移植 kidney transplantation': 'alive',
          '腎功能恢復 recovery of kidney function': 'alive',
          '放棄 withdrawal': 'dead'}

dep = OC[OC['是否離開本院'] == 1].copy()
dep['route_en']  = dep['離開途徑'].map(EN)
dep['assumption'] = dep['離開途徑'].map(ASSUME).fillna('stayer rate')
assert dep.route_en.notna().all(), dep.loc[dep.route_en.isna(), '離開途徑'].unique()

rows = []
for route, g in dep.groupby('離開途徑', sort=False):
    n_test = int(g.analysis_id.isin(test_ids).sum())
    rows.append(dict(route=EN[route], n_cohort=len(g),
                     pct_cohort=100*len(g)/len(dep),
                     n_test_patients=n_test,
                     recorded_deaths=int(g['is_death'].sum()),
                     assumption=ASSUME.get(route, 'stayer rate')))
S5 = pd.DataFrame(rows).sort_values('n_cohort', ascending=False).reset_index(drop=True)
S5.to_csv(TAB/'S5_departure_routes.csv', index=False)

print('=' * 96)
print(f'離開本院者 {len(dep):,} 人（全體 {len(OC):,} 人）· 其中出現在測試集 '
      f'{int(dep.analysis_id.isin(test_ids).sum())} 人（測試集共 {len(test_ids)} 人）')
print('=' * 96)
print(S5.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

# route_informed 與 matched 的實際差距(改由 S1/S2 讀取,不再另算)
import sys; sys.path.insert(0, str(Path(__file__).parent))
S1 = pd.read_csv(TAB/'S1_censoring_sensitivity.csv').set_index('scenario')
S2 = pd.read_csv(TAB/'S2_sensitivity_bootstrap.csv')
_d = lambda sc, m: float(S2[(S2.scenario==sc)&(S2.model==m)].delta.iloc[0])
cmp_rows = [dict(model=m,
                 c_matched=float(S1.loc['matched_rate', f'c_{m}']),
                 c_route=float(S1.loc['route_informed', f'c_{m}']),
                 c_diff=float(S1.loc['route_informed', f'c_{m}'] - S1.loc['matched_rate', f'c_{m}']),
                 delta_matched=(_d('matched_rate', m) if m != 'current_value_cox' else float('nan')),
                 delta_route=(_d('route_informed', m) if m != 'current_value_cox' else float('nan')))
            for m in ('current_value_cox','trajectory_cox','random_survival_forest','gradient_boosted_survival')]
CMP = pd.DataFrame(cmp_rows)
CMP['delta_diff'] = CMP.delta_route - CMP.delta_matched
CMP.to_csv(TAB/'S5_route_vs_matched.csv', index=False)
print('\n' + '-'*96)
print('route-informed 對 matched:途徑資訊改變了多少')
print('-'*96)
print(CMP.to_string(index=False))
print(f'\n  絕對 C 最大差距 {CMP.c_diff.abs().max():.4f}')
print(f'  ΔC 最大差距     {CMP.delta_diff.abs().max():.5f}')

# route_informed 的病人層級指定
assign = dep[['analysis_id', 'assumption']].copy()
assign['route_en'] = dep['route_en'].values
assign = assign[assign.analysis_id.isin(test_ids)]
assign.to_csv(TAB/'S5_route_assignment.csv', index=False)

print('\n' + '-' * 96)
print('route_informed 情境的指定（僅測試集病人）')
print('-' * 96)
print(assign.assumption.value_counts().to_string())

n_dead_route  = int((assign.assumption == 'dead').sum())
n_alive_route = int((assign.assumption == 'alive').sum())
n_stoch       = int((assign.assumption == 'stayer rate').sum())
print(f'\n  假定死亡 {n_dead_route} 人 · 假定存活 {n_alive_route} 人 · 依留院者事件率隨機指定 {n_stoch} 人')
print(f'\n寫出 {TAB}/S5_departure_routes.csv 與 S5_route_assignment.csv')

# 斷言:轉院是主要途徑,且移植者未被當成死亡
assert S5.iloc[0].route.startswith('Transfer'), S5.iloc[0].route
assert S5.iloc[0].n_cohort / S5.n_cohort.sum() > 0.8
assert (S5.loc[S5.route == 'Kidney transplantation', 'assumption'] == 'alive').all()
