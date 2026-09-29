# -*- coding: utf-8 -*-
"""
設限情境的單一定義來源。01、02、09 全部從這裡取資料與指派抽樣。

會有這支模組，是因為交叉審查抓到:同一個 matched 情境，Figure 3A／Table S2 用 40 次
抽樣、Table 3 用另外 200 次，兩者相減對不起來。現在所有腳本共用同一組抽樣。

三個修正一併在此:

1. **指派抽樣只產生一次**，寫進 `S0_scenario_draws.csv`，任何腳本重跑都讀同一份。
2. **matched 情境改用病人層級的死亡比例**。原本用的是未離院 landmark 的事件比例
   （656/1,062 = 0.618），那是「病歷筆數」的比例，不是「人」的比例；同一位死者會讓
   他過去每一年的 landmark 都記為事件。改用未離院**病人**中曾死亡的比例。
3. **情境同時套用到發展組**，因為情境是對世界的一個假設，不是只對測試集的操作。
   校準的 IPCW 權重因此依情境重估，而不是三個情境共用原始的設限分布。
   發展組的離院病人與測試組不重疊（依病人切分），所以**必須有自己的一組指派**；
   第一版把測試組的 id 套到發展組上，等於什麼都沒改，第二輪交叉審查抓到。
"""
import os
import numpy as np, pandas as pd
from pathlib import Path

DV1  = Path(os.environ.get('LOCKED_DIR', 'locked_model_outputs'))
TAB  = Path(os.environ.get('RESULTS_DIR', 'results'))
SEED, N_DRAWS = 20260903, 200
H3 = 1095.75
MODELS = ['current_value_cox', 'trajectory_cox', 'random_survival_forest',
          'gradient_boosted_survival']
NICE = {'current_value_cox': 'Current-value Cox', 'trajectory_cox': 'Trajectory Cox',
        'random_survival_forest': 'Random survival forest',
        'gradient_boosted_survival': 'Gradient-boosted survival'}
REF = 'current_value_cox'

_PR = pd.read_csv(DV1/'05_results'/'tables'/'dynamic_survival_v1_predictions.csv')
_OC = pd.read_csv(DV1/'02_longitudinal_data'/'outcomes.csv')
_LM = pd.read_csv(DV1/'02_longitudinal_data'/'dynamic_landmarks.csv', low_memory=False)

DEPARTED = set(_OC.loc[_OC['是否離開本院'] == 1, 'analysis_id'])
ROUTE    = _OC.set_index('analysis_id')['離開途徑']

test = _PR[_PR.model == REF].reset_index(drop=True)
_key = lambda d: d.set_index(['analysis_id', 'landmark_year']).index
dev  = _LM[~_key(_LM).isin(_key(test))].reset_index(drop=True)

# ---- 測試集
pid   = test.analysis_id.to_numpy()
tm    = test.target_duration_days.to_numpy(float)
ev0   = test.target_event.to_numpy().astype(bool)
dep   = test.analysis_id.isin(DEPARTED).to_numpy()
elig3 = test.calendar_eligible_3y.to_numpy().astype(bool)
risk      = {m: _PR[_PR.model == m].reset_index(drop=True).risk_score.to_numpy() for m in MODELS}
risk3     = {m: _PR[_PR.model == m].reset_index(drop=True)['risk_1095.75'].to_numpy() for m in MODELS}
for m in MODELS:                                  # 逐列驗證 landmark 順序一致
    _g = _PR[_PR.model == m].reset_index(drop=True)
    assert (_g.analysis_id.to_numpy() == pid).all()
    assert (_g.landmark_year.to_numpy() == test.landmark_year.to_numpy()).all()

# ---- 發展組（情境同樣套用,供重估設限分布用）
dev_pid = dev.analysis_id.to_numpy()
dev_tm  = dev.target_duration_days.to_numpy(float)
dev_ev0 = dev.target_event.to_numpy().astype(bool)
dev_dep = dev.analysis_id.isin(DEPARTED).to_numpy()

groups   = {p: np.flatnonzero(pid == p) for p in np.unique(pid)}
patients = np.array(sorted(groups))

# ---- matched 情境的目標比例:病人層級
_stay_pt   = test.loc[~dep].groupby('analysis_id').target_event.max()
RATE_PATIENT  = float(_stay_pt.mean())                 # 未離院病人中曾死亡的比例
RATE_LANDMARK = float(ev0[~dep].mean())                # 舊版用的 landmark 比例,保留供對照
DEP_IDS = np.array(sorted(set(pid[dep])))
DEV_DEP_IDS = np.array(sorted(set(dev_pid[dev_dep])))  # 發展組自己的離院病人,與測試組不重疊
assert len(set(DEP_IDS) & set(DEV_DEP_IDS)) == 0
N_MATCHED = int(round(RATE_PATIENT * len(DEP_IDS)))

# ---- 途徑導向的指派（腎移植與腎功能恢復假定存活,放棄透析假定死亡,其餘隨機）
_EN = {'轉院 transfer to another unit': 'Transfer to another dialysis unit',
       '未見於最終年報 absent from the final annual report': 'Absent from the final annual report',
       '腎移植 kidney transplantation': 'Kidney transplantation',
       '轉腹膜透析 change to peritoneal dialysis': 'Change to peritoneal dialysis',
       '狀態碼4(字典未列) status code 4': 'Status code 4 (not in the data dictionary)',
       '腎功能恢復 recovery of kidney function': 'Recovery of kidney function',
       '放棄 withdrawal': 'Withdrawal from dialysis',
       '不明原因退出 exit, reason not stated': 'Exit, reason not stated'}
_ASSUME = {'腎移植 kidney transplantation': 'alive',
           '腎功能恢復 recovery of kidney function': 'alive',
           '放棄 withdrawal': 'dead'}
ROUTE_EN     = {p: _EN[ROUTE[p]] for p in DEP_IDS}
ROUTE_ASSUME = {p: _ASSUME.get(ROUTE[p], 'stayer rate') for p in DEP_IDS}
DEV_ROUTE_ASSUME = {p: _ASSUME.get(ROUTE[p], 'stayer rate') for p in DEV_DEP_IDS}


def _arm_draws(rng, ids, assume, arm):
    """對一個 arm 的離院病人產生各情境的指派。比例是對世界的假設,兩個 arm 相同。"""
    idx = {p: i for i, p in enumerate(ids)}
    rows = []
    for name, frac in (('fraction_0.25', .25), ('fraction_0.50', .50),
                       ('fraction_0.75', .75), ('matched_rate', RATE_PATIENT)):
        k = int(round(frac*len(ids)))
        for d in range(N_DRAWS):
            pick = rng.choice(ids, size=k, replace=False)
            mask = np.zeros(len(ids), dtype=int); mask[[idx[p] for p in pick]] = 1
            rows.append(dict(arm=arm, scenario=name, draw=d, mask=''.join(map(str, mask))))
    stoch = np.array([p for p in ids if assume[p] == 'stayer rate'])
    forced = [p for p in ids if assume[p] == 'dead']
    k = int(round(RATE_PATIENT * len(stoch)))
    for d in range(N_DRAWS):
        pick = set(rng.choice(stoch, size=k, replace=False)) | set(forced)
        mask = np.zeros(len(ids), dtype=int); mask[[idx[p] for p in pick]] = 1
        rows.append(dict(arm=arm, scenario='route_informed', draw=d, mask=''.join(map(str, mask))))
    return rows


def _build_draws() -> pd.DataFrame:
    """兩個 arm 各自產生指派。發展組的離院病人與測試組不重疊,所以不能共用 id。"""
    rng = np.random.default_rng(SEED)
    rows  = _arm_draws(rng, DEP_IDS, ROUTE_ASSUME, 'test')
    rows += _arm_draws(rng, DEV_DEP_IDS, DEV_ROUTE_ASSUME, 'dev')
    return pd.DataFrame(rows)


_PATH = TAB/'S0_scenario_draws.csv'
if _PATH.exists():
    DRAWS = pd.read_csv(_PATH, dtype={'mask': str})
else:
    TAB.mkdir(parents=True, exist_ok=True)
    DRAWS = _build_draws(); DRAWS.to_csv(_PATH, index=False)
_DRAW_LOOKUP = {(r.arm, r.scenario, r.draw): np.array(list(r.mask), dtype=int).astype(bool)
                for r in DRAWS.itertuples()}
_IDS = {'test': DEP_IDS, 'dev': DEV_DEP_IDS}

# 決定性情境
DETERMINISTIC = {'primary': lambda n: np.zeros(n, dtype=bool),
                 'worst':   lambda n: np.ones(n, dtype=bool)}
STOCHASTIC = ['fraction_0.25', 'fraction_0.50', 'matched_rate', 'fraction_0.75', 'route_informed']
ALL_SCENARIOS = ['primary', 'fraction_0.25', 'fraction_0.50', 'matched_rate',
                 'fraction_0.75', 'route_informed', 'worst']
LABEL = {'primary': 'None (as analysed)', 'fraction_0.25': '25%', 'fraction_0.50': '50%',
         'matched_rate': f'{100*RATE_PATIENT:.1f}% (rate among stayers)',
         'fraction_0.75': '75%', 'route_informed': 'Route-informed',
         'worst': '100%'}


def dead_patients(scenario: str, draw: int = 0, arm: str = 'test') -> np.ndarray:
    """回傳該 arm 中被指派為死亡的離院病人 id。"""
    ids = _IDS[arm]
    m = (DETERMINISTIC[scenario](len(ids)) if scenario in DETERMINISTIC
         else _DRAW_LOOKUP[(arm, scenario, draw)])
    return ids[m]


def events(scenario: str, draw: int = 0, arm: str = 'test') -> np.ndarray:
    """情境下的事件指標。arm='dev' 套用發展組自己的指派,供重估設限分布。"""
    ids = dead_patients(scenario, draw, arm)
    if arm == 'test':
        return ev0 | (dep & np.isin(pid, ids))
    return dev_ev0 | (dev_dep & np.isin(dev_pid, ids))


def n_draws(scenario: str) -> int:
    return 1 if scenario in DETERMINISTIC else N_DRAWS
