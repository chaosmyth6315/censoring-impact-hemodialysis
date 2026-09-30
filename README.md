# Censoring impact in time-updated mortality prediction for hemodialysis

Code and aggregate results for:

> Fang YW, Huang YC, Wang JT, Tsai MH. How much is the non-informative censoring assumption worth? A fixed-model
> sensitivity analysis of time-updated mortality prediction in hemodialysis. (Under review.)

## Contents

| Folder | Content |
|---|---|
| `censoring_impact/` | A tool that runs the same analysis on your own data: hold fitted models fixed, reassign a share of patients who left follow-up as deaths at departure, and see how discrimination and calibration move. See `censoring_impact` usage below. |
| `web/` | `censoring_impact.html`: the same tool as a single web page, online at <https://chaosmyth6315.github.io/censoring-impact-hemodialysis/web/censoring_impact.html> or downloaded and opened in a browser. Choose your CSV and run; nothing is uploaded, and the downloaded page works offline. |
| `results/` | Aggregate result tables from which every number in the paper is computed (no patient-level data). |
| `figures/` | `redraw_from_results.py` redraws Figures 2 and 3 from `results/` alone. |
| `analysis/` | The analysis scripts used for the paper. They read the locked model outputs, which contain patient-level data and are not distributed; set `LOCKED_DIR` and `RESULTS_DIR` to run them where the data are available. Comments are in Chinese. |
| `tests/`, `examples/` | Unit tests of the tool, a check of the web version against the Python tool, and a synthetic example data set. |

## Using the tool on your own data

Without installing anything: open <https://chaosmyth6315.github.io/censoring-impact-hemodialysis/web/censoring_impact.html>, or download `web/censoring_impact.html` and open it in a browser.
Everything is computed in your browser; your data are not uploaded.

With Python:

```bash
pip install -r requirements.txt
python -m censoring_impact --data evaluation.csv --id patient_id --time time --event event --departed departed \
    --model "ModelA=risk_a" --model "ModelB=risk_b" --reference ModelA \
    --risk-at-horizon "ModelA=risk3y_a" --risk-at-horizon "ModelB=risk3y_b" --horizon 3 --out results_mine/
```

One row per evaluation unit (a patient or a patient-period); `departed` marks patients who left follow-up. Your data
stay on your computer. Try it first with `python examples/make_example_data.py` and the command in that file.
The tool reproduces every number in the paper from the study's inputs; the reproduction test needs the
non-distributed data and is kept by the authors.

## Results files

| File | Content |
|---|---|
| `results/S1_censoring_sensitivity.csv` | Harrell's C per model under each scenario (all pairs; between-patient pairs for three scenarios) |
| `results/S2_sensitivity_bootstrap.csv` | Paired differences in C against current-value Cox, patient-clustered bootstrap intervals |
| `results/S4_calibration_sensitivity.csv` | Observed-to-expected ratio and calibration slope at 3 years under each scenario |
| `results/S4_calibration_bins.csv` | Calibration by deciles of predicted risk (Figure 3A) |
| `results/S4_oe_crossing.csv` | Interpolated share of departed patients at which O:E crosses 1 |
| `results/S5_departure_routes.csv` | Routes out of the unit (whole cohort) |
| `results/S5_route_vs_matched.csv` | Route-informed against stayer-rate scenario |
| `results/S6_departure_predicted_risk.csv` | Predicted 3-year risk of departed and remaining test landmarks |
| `results/S7_follow_up.csv` | Follow-up from the landmark |
| `results/S8_fixed_set_slope.csv` | Calibration slope on the landmarks evaluable in every scenario |
| `results/S9_ipcw_calendar_check.csv` | Calibration slope with censoring weights from calendar-eligible development landmarks |
| `results/Table_1_landmark_baseline.csv` | Table 1 (characteristics of development and test landmarks) |
| `results/Table_S1_selection_rule.csv` | Supplementary Table S1 (pre-specified replacement rule) |
| `results/Table_3_sensitivity.csv` | Table 3 panel B (differences in C, formatted) |

Scenario names in the files: `primary` = analysis as conducted; `fraction_0.25` … `fraction_0.75` = that share of
departed patients assumed to have died at departure; `matched_rate` = the stayer rate (65.5%); `route_informed`;
`worst` = all departed patients assumed to have died. Models: current-value Cox (reference), trajectory Cox, random
survival forest, gradient-boosted survival.

## Data availability

The patient-level data are not publicly available because the records remain re-identifiable (single centre, exact
durations). A de-identified analysis data set is available from the corresponding author on reasonable request,
subject to approval by the data custodian and the institutional ethics review board (IRB 20260606R, Shin-Kong Wu
Ho-Su Memorial Hospital). Excluded from this repository for that reason: `S0_scenario_draws.csv` (per-patient assignment masks), `S5_route_assignment.csv` (patient identifiers), `S3_design_comparison.csv` (analysis not part of this paper).

## License

Code: MIT (`LICENSE`). Aggregate results: CC BY 4.0 (`LICENSE-results.md`).

## Contact

Ming-Hsien Tsai, Division of Nephrology, Shin-Kong Wu Ho-Su Memorial Hospital, Taipei, Taiwan. m007358@ms.skh.org.tw
