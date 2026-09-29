"""
Redraws Figures 2 and 3 of the paper from the aggregate results in ../results only (no patient-level data).
Run:  python figures/redraw_from_results.py
The plotting functions are taken unchanged from the authors' figure script; figures are written as PNG.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE.parent/'results', HERE
W = 178/25.4
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': .6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 6.5})
BLACK, GREY, LGREY = '#000000', '#7A7A7A', '#C8C8C8'
S1 = pd.read_csv(RES/'S1_censoring_sensitivity.csv')
S2 = pd.read_csv(RES/'S2_sensitivity_bootstrap.csv')
S4 = pd.read_csv(RES/'S4_calibration_sensitivity.csv')
B4 = pd.read_csv(RES/'S4_calibration_bins.csv')
CROSS = pd.read_csv(RES/'S4_oe_crossing.csv')
MODELS = ['current_value_cox', 'trajectory_cox', 'random_survival_forest', 'gradient_boosted_survival']
NICE = {'current_value_cox': 'Current-value Cox', 'trajectory_cox': 'Trajectory Cox',
        'random_survival_forest': 'Random survival forest', 'gradient_boosted_survival': 'Gradient-boosted survival'}
MK = {'current_value_cox': ('o', 'white'), 'trajectory_cox': ('o', BLACK),
      'random_survival_forest': ('s', GREY), 'gradient_boosted_survival': ('D', BLACK)}
QA = []
def qa(f, m): QA.append((f, m))
def save_tiff(fig, path, fid, panels):
    out = FIG/(fid.replace(' ', '_') + '.png'); fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out.name)


def fig2():
    """設限敏感度:A 絕對 C 隨假設死亡比例;B ΔC 三情境的配對區間。"""
    fid = 'Figure 2'
    ORDER = ['primary','fraction_0.25','fraction_0.50','matched_rate','fraction_0.75','worst']
    f, (a, b) = plt.subplots(1, 2, figsize=(W, W*0.42), gridspec_kw=dict(wspace=.38, width_ratios=[1.15, 1]))
    f.subplots_adjust(left=.09, right=.985, bottom=.20, top=.90, wspace=.38)
    g = S1.set_index('scenario').loc[ORDER]
    xs = np.array([0, .25, .50, float(S4[S4.scenario=='matched_rate'].iloc[0].label.split('%')[0])/100, .75, 1.0])
    for m in MODELS:
        mk, fc = MK[m]
        a.plot(xs, g[f'c_{m}'], ls='-', lw=.7, color=BLACK, marker=mk, ms=4, mfc=fc, mec=BLACK, mew=.7, zorder=3)
    a.set_xticks([0, .25, .5, .75, 1]); a.set_xticklabels(['0', '25%', '50%', '75%', '100%']); a.set_xlim(-.06, 1.06)
    a.axvline(xs[3], ls=(0, (1, 2)), lw=.6, color=GREY, zorder=1)
    _y0 = a.get_ylim()[0]; a.text(xs[3] - .015, _y0 + .0012, 'stayer rate (65.5%)', fontsize=6.2, color='#404040', va='bottom', ha='right')
    a.set_xlabel('Fraction of departed patients assumed to have died'); a.set_ylabel('Harrell C-index')
    a.text(-.17, 1.04, 'A', transform=a.transAxes, fontsize=9, fontweight='bold')
    a.legend(handles=[Line2D([],[],marker=MK[m][0],ls='-',lw=.7,color=BLACK,mfc=MK[m][1],mec=BLACK,mew=.7,ms=4,label=NICE[m])
                      for m in MODELS], loc='upper right', handletextpad=.4, labelspacing=.3)
    nc = [m for m in MODELS if m != 'current_value_cox']
    SC3 = (('primary','white'), ('matched_rate',GREY), ('worst',BLACK))
    b.axvline(0, color=BLACK, lw=.6, zorder=2)
    for i, m in enumerate(nc):
        for k, (sc, fc) in enumerate(SC3):
            r = S2[(S2.scenario==sc)&(S2.model==m)].iloc[0]; y = i + (.22 - .22*k)
            b.plot([r.ci_low, r.ci_high],[y,y], color=BLACK, lw=.7, zorder=2)
            for x in (r.ci_low, r.ci_high): b.plot([x,x],[y-.08,y+.08], color=BLACK, lw=.7)
            b.plot([r.delta],[y], ls='', marker='o', ms=4.4, mfc=fc, mec=BLACK, mew=.7, zorder=4)
    b.set_yticks(range(len(nc))); b.set_yticklabels([NICE[m].replace(' ','\n',1) for m in nc], fontsize=6.8)
    b.set_ylim(-.6, len(nc)-.4); b.set_xlabel('\u0394 C-index vs current-value Cox')
    b.spines['left'].set_visible(False); b.tick_params(axis='y', length=0)
    b.grid(axis='x', color=LGREY, lw=.4, zorder=0); b.set_axisbelow(True)
    b.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=BLACK,mew=.7,ms=4.4,label='as analyzed'),
                      Line2D([],[],marker='o',ls='',mfc=GREY,mec=BLACK,mew=.7,ms=4.4,label='departures die at the rate of stayers'),
                      Line2D([],[],marker='o',ls='',mfc=BLACK,mec=BLACK,ms=4.4,label='all departures = deaths')],
             loc='lower left', bbox_to_anchor=(-.02,-.04), handletextpad=.4, labelspacing=.28)
    b.text(-.42, 1.04, 'B', transform=b.transAxes, fontsize=9, fontweight='bold')
    if S1.best_model.nunique() != 1: qa(fid, 'best model changes across scenarios')
    for m in ('random_survival_forest','gradient_boosted_survival'):
        g2 = S2[S2.model==m]
        if ((g2.delta > 0) & g2.excludes_zero).any(): qa(fid, f'{m} significantly beats Cox somewhere')
    return save_tiff(f, FIG/'unused_V5.tiff', fid, 2)


def fig3():
    """校準:A 分箱曲線;B O:E 隨假設比例,含 bootstrap 區間與跨越點。"""
    fid = 'Figure 3'
    SC = [('primary', 'white', 'as analyzed (departures censored)'),
          ('matched_rate', GREY, 'departures die at the rate of stayers'),
          ('worst', BLACK, 'all departures assumed deaths')]
    f, (a, b) = plt.subplots(1, 2, figsize=(W, W*0.44), gridspec_kw=dict(width_ratios=[1, 1.05]))
    f.subplots_adjust(left=.085, right=.985, bottom=.21, top=.90, wspace=.34)
    REFM = 'current_value_cox'
    g = B4[B4.model == REFM]; lim = 0
    for sc, fc, _ in SC:
        d = g[g.scenario == sc].sort_values('group')
        a.plot(d.mean_predicted, d.observed, ls='-', lw=.5, color=GREY, zorder=1)
        a.plot(d.mean_predicted, d.observed, ls='', marker='o', ms=4.2, mfc=fc, mec=BLACK, mew=.7, zorder=3)
        lim = max(lim, d.mean_predicted.max(), d.observed.max())
    lim = min(1, lim*1.10)
    a.plot([0, lim], [0, lim], ls=(0, (4, 3)), lw=.6, color=BLACK, zorder=2)
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect('equal', adjustable='box')
    a.set_xlabel('Predicted 3-year risk'); a.set_ylabel('Observed 3-year risk (Kaplan\u2013Meier)')
    a.text(-.20, 1.04, 'A', transform=a.transAxes, fontsize=9, fontweight='bold')
    a.legend(handles=[Line2D([], [], marker='o', ls='', mfc=fc, mec=BLACK, mew=.7, ms=4.2, label=lab)
                      for _, fc, lab in SC], loc='upper left', bbox_to_anchor=(-.02, 1.02),
             handletextpad=.4, labelspacing=.3)
    a.text(.97, .04, 'Current-value Cox\nrisk deciles', transform=a.transAxes, ha='right',
           va='bottom', fontsize=6.2, color='#404040', linespacing=1.3)
    ORDER = ['primary','fraction_0.25','fraction_0.50','matched_rate','fraction_0.75','worst']
    FR = [0, .25, .50, float(S4[S4.scenario=='matched_rate'].iloc[0].label.split('%')[0])/100, .75, 1.0]
    gr = S4[S4.model == REFM].set_index('scenario')
    ys = [float(gr.loc[s].oe) for s in ORDER]
    b.axhline(1, ls=(0, (4, 3)), lw=.6, color=BLACK, zorder=2)
    for s, x in zip(ORDER, FR):
        r = gr.loc[s]
        if np.isfinite(r.get('oe_ci_low', np.nan)):
            b.plot([x, x], [r.oe_ci_low, r.oe_ci_high], color=BLACK, lw=.7, zorder=2)
            for yy in (r.oe_ci_low, r.oe_ci_high): b.plot([x-.012, x+.012],[yy,yy], color=BLACK, lw=.7)
    b.plot(FR, ys, ls='-', lw=.7, color=BLACK, zorder=3)
    for s, x, y in zip(ORDER, FR, ys):
        fc = 'white' if s == 'primary' else (BLACK if s == 'worst' else (GREY if s == 'matched_rate' else '#DDDDDD'))
        b.plot([x], [y], ls='', marker='o', ms=4.4, mfc=fc, mec=BLACK, mew=.7, zorder=4)
    cx = float(CROSS.interpolated_crossing.iloc[0])
    b.axvline(cx, ls=(0,(1,2)), lw=.6, color=GREY, zorder=1)
    b.text(cx+.015, b.get_ylim()[0], f'crossing \u2248 {100*cx:.0f}%', fontsize=6.2,
           color='#404040', va='bottom', rotation=90)
    b.set_xlabel('Fraction of departed patients assumed to have died')
    b.set_ylabel('Observed : expected 3-year risk')
    b.set_xlim(-.06, 1.06)
    b.grid(axis='y', color=LGREY, lw=.4, zorder=0); b.set_axisbelow(True)
    b.text(-.19, 1.04, 'B', transform=b.transAxes, fontsize=9, fontweight='bold')
    if not (gr.loc['primary'].oe < 1 < gr.loc['worst'].oe): qa(fid, 'O:E does not cross 1')
    if not (0 < cx < 1): qa(fid, 'crossing point outside [0,1]')
    return save_tiff(f, FIG/'unused_V3.tiff', fid, 2)

(FIG/'Supplementary').mkdir(parents=True, exist_ok=True)

fig2(); fig3()
print('QA notes:', QA or 'none')
