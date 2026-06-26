# Predictive Maintenance (PdM) — Performance Assessment Project

This repository contains a ready-to-run **Predictive Maintenance performance assessment** project using Python.
It includes:
- `src/evaluate.py`: evaluation script (classification + RUL metrics + plots).
- `data/sample_data.csv`: small synthetic example dataset.
- `report/report_template.md`: markdown report template to adapt for your project.
- `requirements.txt`: python dependencies.
- `notebooks/README_NOTEBOOK.md`: guidance if you want to convert to a notebook.

How to use
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. Run the evaluation script:
   ```bash
   python src/evaluate.py --data data/sample_data.csv --out report
   ```

3. Open `report/metrics_summary.txt` and the generated plots in `report/plots/`.

Files and expected columns in CSV
- `timestamp`: ISO date/time or string (used only for ordering).
- `machine_id`: asset identifier.
- `true_label`: 0 or 1 for failure occurrence within horizon (classification ground truth).
- `pred_score`: model predicted probability (0..1) for failure (classification score).
- `true_rul`: true remaining useful life (hours) — optional, used for RUL metrics.
- `pred_rul`: model predicted RUL (hours) — optional.

This project is a template — replace `data/sample_data.csv` with your real dataset and adapt thresholds/assumptions in `src/evaluate.py`.
