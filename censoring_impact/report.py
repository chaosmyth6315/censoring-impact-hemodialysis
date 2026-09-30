"""Writes result tables, a plain-text summary and two figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {'as_analyzed': 'As analyzed (departures censored)', 'stayer_rate': 'Stayer rate',
         'route_informed': 'Route-informed', 'all_departed_died': 'All departed patients died'}


def _f(x, spec='.3f') -> str:
    return '—' if pd.isna(x) else format(x, spec)


def _label(sc) -> str:
    if sc.kind == 'fraction':
        return f'{100 * sc.fraction:g}% of departed patients died'
    if sc.kind == 'stayer':
        return f'{100 * sc.fraction:.1f}% (stayer rate)'
    return LABEL[sc.name]


def write(res, cfg, out: Path, config_record: dict) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    lab = {s.name: _label(s) for s in res.scenarios}
    files = []
    d = res.discrimination.copy(); d.insert(1, 'label', d.scenario.map(lab))
    d.to_csv(out/'discrimination_by_scenario.csv', index=False); files.append(out/'discrimination_by_scenario.csv')
    res.paired.to_csv(out/'paired_differences.csv', index=False); files.append(out/'paired_differences.csv')
    res.draws.to_csv(out/'assignment_draws.csv', index=False); files.append(out/'assignment_draws.csv')
    if res.calibration is not None:
        c = res.calibration.copy(); c.insert(1, 'label', c.scenario.map(lab))
        c.to_csv(out/'calibration_by_scenario.csv', index=False); files.append(out/'calibration_by_scenario.csv')
    if res.crossing:
        pd.DataFrame([res.crossing]).to_csv(out/'oe_crossing.csv', index=False); files.append(out/'oe_crossing.csv')
    (out/'run_config.json').write_text(json.dumps(dict(config_record, exposure=res.exposure), indent=2, default=str))
    files.append(out/'run_config.json')
    files.append(_summary(res, cfg, lab, out/'summary.md'))
    files += _figures(res, cfg, out)
    return files


def _summary(res, cfg, lab, path: Path) -> Path:
    e, d = res.exposure, res.discrimination.set_index('scenario')
    models = list(cfg.risk)
    first, last = 'as_analyzed', 'all_departed_died'
    L = ['# Censoring-impact summary', '',
         '## Exposure', '',
         f"- {e['rows_from_departed_patients']:,} of {e['rows']:,} evaluation rows ({100 * e['share_rows_departed']:.1f}%) "
         f"come from {e['departed_patients']:,} of {e['patients']:,} patients who left follow-up; "
         f"{e['events_recorded_in_departed_rows']} deaths are recorded in those rows.",
         f"- Among patients who never left, {100 * e['stayer_rate']:.1f}% have a recorded death (the stayer rate); "
         f"at that rate {e['departed_assigned_at_stayer_rate']} departed patients are assigned death at departure.",
         '', '## Discrimination (Harrell C, models held fixed)', '',
         '| Scenario | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for sc, r in d.iterrows():
        L.append(f'| {lab[sc]} | ' + ' | '.join(f"{r[f'c_{m}']:.3f}" for m in models) + ' |')
    falls = {m: d.loc[first, f'c_{m}'] - d.loc[last, f'c_{m}'] for m in models}
    order = [sorted(models, key=lambda m: -r[f'c_{m}']) for _, r in d.iterrows()]
    L += ['', f"- From the analysis as conducted to all departed patients assumed dead, C falls by "
              f"{min(falls.values()):.3f} to {max(falls.values()):.3f} across models.",
          f"- Model ordering by C is {'the same in every scenario' if all(o == order[0] for o in order) else 'NOT the same in every scenario'}."]
    if 'cx_' + models[0] in d.columns:
        cx = [abs(d.loc[s, f'c_{m}'] - d.loc[s, f'cx_{m}']) for s in d.index for m in models if not pd.isna(d.loc[s, f'cx_{m}'])]
        L.append(f'- Restricting C to pairs formed between different patients changes it by at most {max(cx):.4f}.')
    if not res.paired.empty:
        L += ['', f'## Paired differences in C against {cfg.reference} (95% patient-clustered bootstrap interval)', '',
              '| Model | ' + ' | '.join(lab[s] for s in res.paired.scenario.unique()) + ' |',
              '|---|' + '---|' * res.paired.scenario.nunique()]
        for m, g in res.paired.groupby('model', sort=False):
            L.append(f'| {m} | ' + ' | '.join(f"{_f(r.delta, '+.4f')} ({_f(r.ci_low, '+.4f')}, {_f(r.ci_high, '+.4f')})"
                                               for r in g.itertuples()) + ' |')
        flips = [m for m, g in res.paired.groupby('model') if g.excludes_zero.nunique() > 1]
        if flips:
            L.append(f"- Whether the interval excludes zero depends on the assumption for: {', '.join(flips)}.")
    if res.calibration is not None:
        c = res.calibration
        ref = cfg.reference if cfg.reference in c.model.unique() else c.model.unique()[0]
        g = c[c.model == ref].set_index('scenario')
        L += ['', f'## Calibration at the horizon ({ref})', '', '| Scenario | O:E | 95% CI | Slope | 95% CI | Rows for slope |',
              '|---|---|---|---|---|---|']
        for sc, r in g.iterrows():
            ci = lambda q: (f"({r[f'{q}_ci_low']:.3f}, {r[f'{q}_ci_high']:.3f})"
                            if f'{q}_ci_low' in g.columns and not pd.isna(r.get(f'{q}_ci_low')) else '—')
            L.append(f"| {lab[sc]} | {_f(r.oe)} | {ci('oe')} | {_f(r.slope)} | {ci('slope')} | {int(r.n_slope):,} |")
        if res.crossing:
            L.append(f"\n- The point estimate of O:E crosses 1 at about {100 * res.crossing['interpolated_crossing']:.0f}% of "
                     'departed patients assumed to have died at departure (linear interpolation between point estimates).')
        else:
            L.append('\n- The point estimate of O:E does not cross 1 across the scenarios examined.')
    L += ['', '## Reading these results', '',
          '- Nothing is refitted: the models and their predictions are held fixed and only the outcome definition changes.',
          '- Deaths are assigned at the moment of departure and, in the random scenarios, independently of predicted risk. '
          'The scenarios are alternative outcome definitions, not a formal bound on the true performance.',
          '- Intervals cover the resampling of patients and, in random scenarios, which departed patients were assigned '
          'death; they do not cover model fitting or the uncertainty of the stayer rate itself.']
    L += [f'- {x}' for x in res.notes]
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')
    return path


def _figures(res, cfg, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    sc = [s for s in res.scenarios if s.fraction is not None]
    x = np.array([s.fraction for s in sc])
    d = res.discrimination.set_index('scenario')
    stay = next(s for s in sc if s.kind == 'stayer').fraction
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, mk in zip(cfg.risk, 'osD^vP*X'):
        ax.plot(x, [d.loc[s.name, f'c_{m}'] for s in sc], marker=mk, label=m)
    ax.axvline(stay, ls=':', color='grey'); ax.text(stay, ax.get_ylim()[0], f' stayer rate ({100 * stay:.1f}%)', fontsize=8, va='bottom')
    ax.set_xlabel('Fraction of departed patients assumed to have died at departure'); ax.set_ylabel('Harrell C-index')
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%']); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'figure_discrimination.png', dpi=200); plt.close(fig); files.append(out/'figure_discrimination.png')
    if res.calibration is not None:
        c = res.calibration
        ref = cfg.reference if cfg.reference in c.model.unique() else c.model.unique()[0]
        g = c[c.model == ref].set_index('scenario')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, [g.loc[s.name, 'oe'] for s in sc], marker='o', color='black')
        if 'oe_ci_low' in g.columns:
            for s in sc:
                r = g.loc[s.name]
                if not pd.isna(r.get('oe_ci_low')):
                    ax.plot([s.fraction] * 2, [r.oe_ci_low, r.oe_ci_high], color='black', lw=.8)
        ax.axhline(1, ls='--', color='black', lw=.8)
        if res.crossing:
            ax.axvline(res.crossing['interpolated_crossing'], ls=':', color='grey')
        ax.set_xlabel('Fraction of departed patients assumed to have died at departure')
        ax.set_ylabel(f'Observed : expected risk at the horizon ({ref})')
        ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        fig.tight_layout(); fig.savefig(out/'figure_calibration.png', dpi=200); plt.close(fig); files.append(out/'figure_calibration.png')
    return files
