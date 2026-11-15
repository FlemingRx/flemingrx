# FlemingRx app

A Streamlit app for clinicians managing antibiotics for urine infections, using the model
registered by `notebooks/03_train.py`.

Screens, from the sidebar:

- **Dashboard**: the landing page. Today's worklist, cases needing attention, recent activity,
  pending cultures, and the button that starts an assessment. It shows sample data for now
  (written out at the top of `dashboard.py`); there is no patient database yet.
- **Clinical Assessment**: a short form, each antibiotic's predicted chance of working, and the
  pick. Its **Evidence** tab has the trade-off chart from `notebooks/04_recommend.py` and its
  **Limits** tab says what the tool does not do.
- **Coming next**: Patient Encounters, Culture Results, Follow-up and Stewardship Analytics are
  placeholders that say what each will hold.

## How it fits together

| File | What it does |
|---|---|
| `app.py` | The frame around every screen (sidebar, top bar) and the clinical assessment. |
| `dashboard.py` | The dashboard and its sample data. |
| `policy.py` | The recommendation rule: the narrowest antibiotic that clears the bar. |
| `data.py` | Loads the model and tables; turns patient details into model features. |
| `app.yaml` | Databricks Apps configuration. |
| `styles.css`, `.streamlit/config.toml`, `assets/` | The FlemingRx look: brand colours and fonts, card and result-panel styling, the logo. |

Everything works from one `patient` dict of `{feature name: number}`. Today the form fills it in.
Reading a patient from medical records later means adding one function in `data.py` that returns
the same dict; nothing else has to change.

`tests/test_policy.py` checks that `policy.recommend` gives the same picks as the `recommend`
function in notebook 04, so the app applies the rule that was evaluated.

## Run it on Databricks

Run notebooks 01 to 04 first, so the model and the `gold_policy_tradeoff` table exist.

1. In the workspace, create a Databricks App from this `app/` folder.
2. Add a **SQL warehouse** resource to the app and keep the default key `sql-warehouse`
   (`app.yaml` reads the warehouse ID from it).
3. Give the app's service principal:
   - `USE CATALOG` on `workspace` and `USE SCHEMA` on `workspace.armd`
   - `EXECUTE` on the model `workspace.armd.urine_resistance_model`
   - `SELECT` on `workspace.armd.gold_policy_tradeoff`
   - `CAN USE` on the SQL warehouse
4. Deploy.

Deploying as a Databricks App has not been tried yet. Loading the registered model and reading
the table from Databricks has, from a laptop. If the model fails to load with a scikit-learn
version error, update the pin in `requirements.txt` (the comment there says where to find it).

## Run it locally

Install the requirements into your environment first. Without them the app reports that the
model could not be loaded.

```bash
pip install -r app/requirements.txt
```

```bash
cd app && streamlit run app.py
```

Run it from inside `app/` so Streamlit picks up the brand theme in `app/.streamlit/config.toml`
(Streamlit reads that file from the folder it is started in).

The app looks for the model and table in this order:

1. **`MODEL_URI` and `LOCAL_DATA_DIR`**, if set: a saved MLflow model folder (the output of
   `mlflow.pyfunc.save_model`) and a folder holding `gold_policy_tradeoff.parquet`.
2. **`data/app_local/`** in the repo, if it exists: `data/app_local/model/` and
   `data/app_local/tables/`. This folder is git-ignored, so each person needs their own copy.
3. **Databricks**: the registered model in Unity Catalog and the table through a SQL warehouse.
   This needs the Databricks CLI signed in (`databricks auth login`) and
   `DATABRICKS_WAREHOUSE_ID` set to the warehouse ID (the last part of the warehouse's HTTP path
   under SQL Warehouses -> Connection details).

The footer shows which model is in use.

Other settings: `TARGET_SUCCESS` (default `0.90`) sets where the slider starts, and `MODEL_NAME`
picks a different registered model.

### Versions

- **scikit-learn must match the model.** `requirements.txt` pins the version the registered model
  was trained with. A different version cannot open the model.
- **Python does not have to match.** MLflow saves the model's small wrapper class as code for the
  notebook's exact Python version (3.12 on Databricks), and running that under another version
  crashes Python. `data.predict` avoids it by scoring the per-drug models directly.

## Tests

```bash
pip install pytest && pytest tests
```
