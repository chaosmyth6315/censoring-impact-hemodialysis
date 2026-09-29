"""
Writes a small synthetic data set (no real patients) for trying the tool:

    python examples/make_example_data.py
    python -m censoring_impact --data examples/example_evaluation.csv --weights-data examples/example_development.csv \
        --id patient_id --time time --event event --departed departed --route route \
        --model "Cox=risk_cox" --model "Forest=risk_forest" --reference Cox \
        --risk-at-horizon "Cox=risk3y_cox" --risk-at-horizon "Forest=risk3y_forest" --horizon 3 \
        --calibration-eligible horizon_observable --draws 50 --boot 200 --out examples/results

Each patient contributes up to five yearly rows; about a fifth of patients leave follow-up, and their later deaths
are not recorded, exactly the situation the tool is built for.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def cohort(n_patients, seed, prefix):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        frailty = rng.normal()
        leaves = rng.random() < .2
        leave_year = rng.integers(1, 5) if leaves else None
        route = rng.choice(['random', 'random', 'random', 'censored', 'dead']) if leaves else ''
        death_year = rng.exponential(4 / np.exp(.8 * frailty))
        for k in range(rng.integers(1, 6)):
            if death_year <= k:
                break
            end = min(death_year, leave_year if leaves else 99, 6.0)
            if end <= k:
                break
            t = end - k
            died = (death_year <= end) and not (leaves and leave_year <= death_year)
            score = frailty + rng.normal(0, .6)
            rows.append(dict(patient_id=f'{prefix}{i:04d}', landmark=k, time=round(t, 3), event=int(died),
                             departed=int(leaves), route=route, horizon_observable=int(k <= 3),
                             risk_cox=score, risk_forest=score + rng.normal(0, .4),
                             risk3y_cox=1 / (1 + np.exp(-(score - .6))),
                             risk3y_forest=1 / (1 + np.exp(-(score + rng.normal(0, .4) - .6)))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    cohort(300, 1, 'T').to_csv(HERE/'example_evaluation.csv', index=False)
    cohort(900, 2, 'D').to_csv(HERE/'example_development.csv', index=False)
    print('wrote examples/example_evaluation.csv and examples/example_development.csv')
