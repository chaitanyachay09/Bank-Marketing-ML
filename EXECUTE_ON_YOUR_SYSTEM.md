# Run the project on your computer

## Windows — easiest method
1. Install Python 3.11 or newer from https://www.python.org/downloads/.
   During setup, enable **Add Python to PATH**.
2. Extract this ZIP to a normal folder (not inside the ZIP preview).
3. Open the extracted `CSE570_Bank_Marketing_Project` folder.
4. Double-click `RUN_ON_WINDOWS.bat`.
5. Keep the window open. It creates `.venv`, installs `requirements.txt`,
   downloads the UCI dataset (internet required), and runs `src/train.py`.
6. When it says SUCCESS, open the `outputs` folder.
7. Open `analysis_walkthrough.ipynb` in VS Code (with the Python and Jupyter
   extensions) or Jupyter Notebook. Select the `.venv` Python kernel and run
   the cells from top to bottom.
8. Use `docs/COMPLETE_ANALYSIS_AND_DEMONSTRATION_GUIDE.md` to explain each
   analysis step and `docs/REPORT_TEMPLATE.md` to write your report using
   actual generated results.

## Windows — manual terminal method
Open a terminal in the project root and run:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\train.py
```

If `py` is unavailable, replace `py -3` with `python`.

## macOS / Linux
Open Terminal in the project root and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
```

## What you should see
The script prints progress for data download, EDA, statistical tests,
preprocessing, model training/tuning and final evaluation. It writes generated
tables, charts, metrics and the fitted model into `outputs/`.

Important output files include:
- `data_profile.json`
- `missing_value_summary.csv`
- `descriptive_statistics.csv`
- `statistical_tests.json`
- `model_comparison_cv.csv`
- `final_test_metrics.json`
- `permutation_importance.csv`
- `best_model.joblib`
- generated `.png` charts

## Run the analysis notebook
After the training script completes:
1. Open `analysis_walkthrough.ipynb`.
2. Choose the Python interpreter from this project's `.venv`.
3. Run notebook cells in order.
4. The notebook explains the analysis and reads the pipeline results where
   available. The training script is the authoritative end-to-end pipeline.

## If execution fails
- `python` / `py` not found: install Python and enable PATH; reopen terminal.
- Package installation fails: check internet and rerun the pip install command.
- UCI download fails: check internet/proxy access and rerun the script.
- PowerShell activation is blocked: use the manual commands above, which call
  the virtual-environment Python directly and do not require activation.
- Do not make up metrics. Use only values generated in your `outputs/` folder.

The first run requires internet access to install packages and fetch the UCI data.
Training time depends on your computer. Keep the terminal open until it finishes.

Run the Streamlit Application

After the training pipeline completes successfully, start the interactive prediction application:

streamlit run app.py

Streamlit will provide a local URL, generally:

http://localhost:8501

Open this URL in your browser.

Application Functionality

The application allows users to:

Enter customer demographic information.

Select employment and marital status.

Provide financial and banking details.

Enter campaign-related information.

Submit customer details for prediction.

View the predicted term deposit subscription result.

View the estimated subscription probability when available.

The application loads the trained model from:

outputs/best_model.joblib

Therefore, complete model training before launching the application.
