# Dynamical Systems Analysis of Early-Stage Thermal Runaway in Lithium-Ion Batteries

This repository contains a selected subset of code, final figures, and compact result summaries from my MSc thesis project at Uppsala University. The project studied early-stage thermal runaway in lithium-ion batteries using nonlinear ordinary differential equation modelling, dynamical-systems analysis, numerical simulation, and scientific computing.

This is a curated public repository for academic and PhD-application purposes. It is not a full backup of the original thesis working directory and does not include draft documents, private forms, reference-paper PDFs, large raw sweep arrays, or intermediate exploratory outputs.

The final thesis PDF is included as [`thesis.pdf`](thesis.pdf).

## Thesis Context

The thesis investigated a two-dimensional nonlinear ODE model for battery temperature and state of charge. The analysis examined how internal short-circuit resistance, heat-transfer strength, ambient temperature, and external current affect early-stage thermal runaway behavior.

## Code Overview

The selected scripts in `code/` are representative final analysis scripts from the thesis work:

1. [`01_heat_generation_timescale_interpretation.py`](code/01_heat_generation_timescale_interpretation.py): evaluates heat-generation terms and their temperature sensitivity.
2. [`02_timescale_ratio_parameter_partition.py`](code/02_timescale_ratio_parameter_partition.py): analyzes parameter-space regions based on the time-scale ratio.
3. [`03_critical_manifold_phase_plane.py`](code/03_critical_manifold_phase_plane.py): computes critical-manifold and phase-plane trajectory visualizations.
4. [`04_critical_manifold_and_jacobian_field.py`](code/04_critical_manifold_and_jacobian_field.py): combines critical-manifold analysis with local Jacobian interpretation.
5. [`05_jacobian_indicator_analysis.py`](code/05_jacobian_indicator_analysis.py): evaluates local thermal stability indicators along representative trajectories.
6. [`06_thermal_boundary_timescale_analysis.py`](code/06_thermal_boundary_timescale_analysis.py): analyzes thermal threshold boundaries and epsilon/time-scale ratios.
7. [`07_ambient_temperature_sensitivity.py`](code/07_ambient_temperature_sensitivity.py): studies sensitivity to ambient temperature.
8. [`08_external_current_sensitivity.py`](code/08_external_current_sensitivity.py): studies sensitivity to external current.

Some scripts may expect generated `.npz` sweep data from the original thesis workflow. Large raw sweep arrays are intentionally excluded from this public repository.

## Final Thesis Figures

The final thesis figures are provided as PDF files in `figures/pdf/`, organized in thesis order.

### Heat Generation and Model Interpretation

- **Figure 1 - Chemical heat sensitivity vs. cooling coefficients**: Compares the temperature sensitivity of chemical heat generation with representative cooling coefficients. [PDF](figures/pdf/fig01_chemical_heat_sensitivity_vs_cooling.pdf)

### Time-Scale Ratio and Parameter Partition

- **Figure 2 - Time-scale ratio parameter partition**: Shows the parameter-space regions defined by the time-scale ratio. [PDF](figures/pdf/fig02_timescale_ratio_parameter_partition.pdf)

### Critical Manifold and Phase-Plane Interpretation

- **Figure 3 - Critical manifold and trajectories**: Shows representative phase-plane trajectories approaching the critical manifold. [PDF](figures/pdf/fig03_critical_manifold_phase_plane_trajectories.pdf)
- **Figure 4 - Critical temperature for fast thermal attraction**: Visualizes the condition where chemical heat sensitivity balances cooling. [PDF](figures/pdf/fig04_fast_thermal_attraction_critical_temperature.pdf)
- **Figure 5 - Strong-separation temperature classification**: Classifies regions inside the strongly separated parameter regime. [PDF](figures/pdf/fig05_strong_separation_temperature_classification.pdf)

### Local Stability / Jacobian Indicator

- **Figure 6 - Stable low-epsilon trajectory**: Shows a representative trajectory with local damping along the simulated path. [PDF](figures/pdf/fig06_jacobian_indicator_stable_low_epsilon.pdf)
- **Figure 7 - Warning-level intermediate-epsilon trajectory**: Shows a trajectory crossing the 70 C warning level while remaining below 120 C. [PDF](figures/pdf/fig07_jacobian_indicator_warning_intermediate_epsilon.pdf)
- **Figure 8 - High-temperature trajectory case**: Shows a stronger internal short-circuit case where the local indicator becomes positive before the cutoff. [PDF](figures/pdf/fig08_jacobian_indicator_high_temperature_case.pdf)

### Thermal Threshold Boundaries and Epsilon Analysis

- **Figure 9 - Epsilon along the 70 C boundary**: Samples the time-scale ratio along the simulated 70 C thermal boundary. [PDF](figures/pdf/fig09_epsilon_along_70C_boundary.pdf)
- **Figure 10 - Epsilon along the 120 C boundary**: Samples the time-scale ratio along the simulated 120 C thermal boundary. [PDF](figures/pdf/fig10_epsilon_along_120C_boundary.pdf)
- **Figure 11 - Thermal boundaries and mean time-scale ratio curves**: Compares simulated thermal boundaries with corresponding mean time-scale ratio curves. [PDF](figures/pdf/fig11_thermal_boundaries_mean_timescale_ratio.pdf)

### Ambient Temperature Sensitivity

- **Figure 12 - Ambient-temperature thermal boundaries**: Compares simulated 70 C and 120 C boundaries under different ambient temperatures. [PDF](figures/pdf/fig12_ambient_temperature_thermal_boundaries.pdf)
- **Figure 13 - Ambient-temperature mean time-scale curves**: Shows mean time-scale ratio curves corresponding to the ambient-temperature boundary cases. [PDF](figures/pdf/fig13_ambient_temperature_mean_timescale_curves.pdf)
- **Figure 14 - 20 C ambient case**: Shows thermal boundaries and mean time-scale ratio curves for the 20 C ambient case. [PDF](figures/pdf/fig14_ambient_20C_boundaries_timescale.pdf)
- **Figure 15 - 60 C ambient case**: Shows thermal boundaries and mean time-scale ratio curves for the 60 C ambient case. [PDF](figures/pdf/fig15_ambient_60C_boundaries_timescale.pdf)

### External Current Sensitivity

- **Figure 16 - External-current thermal boundaries**: Compares simulated thermal boundaries under different external currents at 40 C ambient temperature. [PDF](figures/pdf/fig16_external_current_thermal_boundaries.pdf)
- **Figure 17 - External-current mean time-scale curves**: Shows mean time-scale ratio curves corresponding to the external-current boundary cases. [PDF](figures/pdf/fig17_external_current_mean_timescale_curves.pdf)
- **Figure 18 - 0 A current case**: Shows thermal boundaries and mean time-scale ratio curves for zero external current. [PDF](figures/pdf/fig18_current_0A_boundaries_timescale.pdf)
- **Figure 19 - 2.5 A current case**: Shows thermal boundaries and mean time-scale ratio curves for 2.5 A external current. [PDF](figures/pdf/fig19_current_2p5A_boundaries_timescale.pdf)
- **Figure 20 - 7.5 A current case**: Shows thermal boundaries and mean time-scale ratio curves for 7.5 A external current. [PDF](figures/pdf/fig20_current_7p5A_boundaries_timescale.pdf)
- **Figure 21 - 25 A current case**: Shows thermal boundaries and mean time-scale ratio curves for 25 A external current. [PDF](figures/pdf/fig21_current_25A_boundaries_timescale.pdf)

## Result Summaries

The `results/` folder contains compact CSV and text summaries used to document selected boundary and sensitivity calculations. Large raw `.npz` arrays are excluded.

Included summaries:

1. [`01_thermal_boundary_mean_epsilon_summary.csv`](results/01_thermal_boundary_mean_epsilon_summary.csv)
2. [`02_120C_stopping_boundary_segment_stats.csv`](results/02_120C_stopping_boundary_segment_stats.csv)
3. [`03_ambient_temperature_boundary_epsilon_summary.txt`](results/03_ambient_temperature_boundary_epsilon_summary.txt)
4. [`04_external_current_boundary_epsilon_summary_Tamb40C.txt`](results/04_external_current_boundary_epsilon_summary_Tamb40C.txt)
5. [`05_external_current_SOC_depletion_summary_Tamb40C.txt`](results/05_external_current_SOC_depletion_summary_Tamb40C.txt)

## Repository Structure

```text
thermal-runaway-dynamical-systems-public-upload/
├── README.md
├── requirements.txt
├── .gitignore
├── thesis.pdf
├── code/
│   └── selected thesis analysis scripts
├── figures/
│   └── pdf/
│       └── final thesis figure PDFs in thesis order
└── results/
    └── compact result summaries
```

## How to Run

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Then run individual scripts from the repository root or from the `code/` directory, for example:

```bash
python code/06_thermal_boundary_timescale_analysis.py
python code/07_ambient_temperature_sensitivity.py
python code/08_external_current_sensitivity.py
python code/05_jacobian_indicator_analysis.py
```

## Requirements

The selected scripts use NumPy, SciPy, Matplotlib, and Pandas. Install them with:

```bash
pip install -r requirements.txt
```

## Note

This repository contains selected code and final figures from an MSc thesis project. It should be read as a concise research-code portfolio rather than a complete reproduction archive.
