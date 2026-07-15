# Phone re-run checklist (via remote desktop)

You are looking at the LAPTOP screen from your phone (Chrome Remote Desktop).
Do these in order. Total time ~20-40 min for the quick pass.

## 1. Open MATLAB, run one command
In the MATLAB Command Window, type:

    cd 'C:\Users\Ahmed\OneDrive\Desktop\try for 5.5'
    run_thesis_rerun

That runs, in order:
  - day8_real_field_perception       (real-field GT recall; expect recall=1.00,
                                      coverage=95%, missed_zones=0)
  - day6 lawnmower  @ 5 Hz            (5 controllers vs real GT)
  - day6 circular   @ 5 Hz           (5 controllers vs real GT)
  - refreshes LaTeX tables + day7 thesis package (Python)

(For the slow Monte Carlo table, run `run_thesis_rerun(10)` overnight instead — the
report uses a 10-seed campaign.)

## 2. What "success" looks like
As each Day-6 run finishes it prints a recall line PER controller, e.g.:

    Recall by controller (agricultural):
      PID    recall=0.xx  coverage=xx.x%  false_alarm=0.xxxxx  latency=x.xxs
      LQR    recall=0.xx  ...
      Hinf   ...
      MPC    ...
      SMC    ...

CHECK:  recall is NON-ZERO  and  DIFFERS across the 5 controllers.
        (If every recall is exactly 0 -> real GT didn't load; tell me.)
        (If every recall is identical 1.00 -> field too dense; tell me.)

## 3. If the Python refresh step fails
The MATLAB run prints the command to run yourself. In a PowerShell terminal
(also on the remote screen), from the same folder:

    python analysis/thesis_analysis.py
    python scripts/day7_finalize_thesis_package.py

## 4. Then send me
Copy the per-controller recall lines from step 2 back to me and I'll help you
read them and finish the thesis table.

## Files that get refreshed
  results/analysis/thesis_tables.tex
  results/perception_logs/day7_thesis_package/
  results/perception_logs/day6_controller_comparison/day6_controller_metrics.csv
  results/perception_logs/day6_circular/day6_controller_metrics.csv
  results/perception_logs/day8_real_field/real_field_metrics.csv
