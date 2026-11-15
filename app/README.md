# FlemingRx app

A Streamlit app for clinicians managing antibiotics for urine infections, using the model
registered by `notebooks/03_train.py`.

Each clinician signs in with their own username and password and sees only the patients
assigned to them. Screens, from the sidebar:

- **Dashboard**: the landing page, read from the database for the clinician's patients:
  today's counts, patients needing reassessment, cultures awaiting a result or review, and a
  7-day stewardship snapshot. Every count, row and "See all" opens the screen behind it.
- **Clinical Assessment**: opens on the list of the clinician's patients. Selecting one shows
  their details from the database, their severity level and the bar it sets, four pre-checks (symptoms, allergies, kidney function,
  pregnancy), each antibiotic's predicted chance of working, and the pick. The clinician accepts
  the recommendation, overrides it with a reason, or decides not to treat; the first two are
  saved as a prescription, and if the patient already has one it is shown instead. Each
  recommendation and what was decided is written to the assessment log. Its **Evidence** tab has the trade-off chart from
  `notebooks/04_recommend.py` and its **Limits** tab says what the tool does not do.
- **Culture Results**: four counts (pending, ready for review, resistance mismatches, reviewed)
  and the list of the clinician's cultures. Selecting one shows the patient's details, the
  culture, the lab's susceptibility results, how the current prescription compares with them,
  and a form to record the clinician's review. A mismatch is a prompt to reassess; nothing here
  changes a prescription.
- **Follow-up**: four counts (needs reassessment, awaiting follow-up, adverse reactions,
  resolved) and a queue of the clinician's patients that can be searched by patient ID and
  filtered by status. Selecting a patient shows their details, the original treatment, why they
  are in the queue, what has been documented since, a summary with a suggested next step, and a
  form to record the clinician's reassessment. The summary is fixed wording picked by rules from
  what is on record, not written by a model.
- **Stewardship Analytics**: totals across the clinician's patients for a chosen reporting
  period: antibiotics prescribed, how often the model's recommendation was accepted, culture
  and resistance findings, and follow-up outcomes. Every figure is counted from the database;
  where something isn't recorded yet the page says so. Buttons open the other screens filtered
  to the cases behind a figure.

## How it fits together

| File | What it does |
|---|---|
| `app.py` | The frame around every screen (sign-in, sidebar, top bar) and the clinical assessment. |
| `dashboard.py` | The dashboard. |
| `culture_results.py` | The culture worklist and the page for one culture. |
| `follow_up.py` | The follow-up queue and the page for one patient's follow-up. |
| `analytics.py` | The stewardship analytics page and the counting behind it. |
| `ui.py` | Building blocks the screens share: cards and the read-only patient record. |
| `policy.py` | The recommendation rule: the narrowest antibiotic that clears the bar. Also a usual adult regimen for each antibiotic. |
| `prechecks.py` | The four pre-checks. Simple rules for the demo, not reviewed by a clinician. |
| `data.py` | Loads the model and tables; turns a patient's database row into model features. |
| `db.py` | The database: clinicians, patients, who looks after whom, and password checking. |
| `seed.py` | The fabricated demo clinicians, patients, prescriptions, cultures and follow-ups, loaded into an empty database. |
| `app.yaml` | Databricks Apps configuration. |
| `styles.css`, `.streamlit/config.toml`, `assets/` | The FlemingRx look: brand colours and fonts, card and result-panel styling, the logo. |

Everything works from one `patient` dict of `{feature name: number}`, built by
`data.patient_features` from the patient's row in the database. Reading patients from a hospital's
records instead means filling the `patients` table from them; nothing else has to change.

## The database

Eight tables (defined in `db.py`):

| Table | What it holds |
|---|---|
| `clinicians` | One row per login: username, name, role, and a salted hash of the password. |
| `patients` | One row per patient: patient number and the details the model uses (age band, sex, care setting, nursing home contact, resistance history, antibiotic exposure, earlier organisms, procedures), plus what the pre-checks use (symptoms, allergies, eGFR, pregnancy) and the vital signs that set the severity level. |
| `clinician_patients` | One row per clinician and patient pair, so a patient can have several clinicians. |
| `prescriptions` | What the model recommended for a patient and what the clinician prescribed: drug, dose, frequency, duration, start date, whether it was an acceptance or an override, and the override reason. |
| `assessments` | The log. One row each time a recommendation is produced for a patient without a prescription: when, by which model, the bar, each antibiotic's predicted chance of working, whether anything cleared the bar, what the pre-checks found, and what the clinician decided (accepted, overridden, or no antibiotic) with the reason. |
| `cultures` | One row per urine culture: specimen, dates, organism, colony count, and the clinician's review (decision, notes, who, when). |
| `susceptibilities` | One row per culture and antibiotic tested: susceptible, intermediate or resistant. |
| `follow_ups` | One row per patient being followed after a prescription: the reason and where it came from, what has been documented since (symptoms, adherence, adverse reactions, notes), the queue status, and the clinician's reassessment. |

The app uses Lakebase (Databricks' managed Postgres), both deployed and on a laptop. The workspace
has a Lakebase project for this, `flemingrx`, holding the tables and demo data in a schema called
`flemingrx`. There is no stored database password: each new connection signs in with a one-hour
token the app asks Databricks for.

- **On a laptop** this needs the Databricks CLI signed in (`databricks auth login`) with an
  account that can use the project. Nothing else has to be set.
- **Without Databricks**, set `DATABASE_URL=sqlite:///flemingrx.db` to use a file on your machine
  instead (git-ignored, created and filled with the demo data on first start).
- **Any other Postgres**: set `DATABASE_URL` (`postgresql://user:password@host:5432/dbname`).
- A different Lakebase compute: set `ENDPOINT_NAME` to its resource name (default in `db.py`).

When deploying as a Databricks App:

1. Add the `flemingrx` project's database to the app as a **Database** resource. This creates a
   Postgres login for the app and sets `PGHOST`, `PGUSER` and the other `PG*` settings for it.
2. The tables were created from a laptop, so they belong to that person's login, not the app's.
   Give the app access by running this in the Lakebase SQL editor, with the app's service
   principal client ID in place of `<client-id>`:

   ```sql
   GRANT USAGE ON SCHEMA flemingrx TO "<client-id>";
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA flemingrx TO "<client-id>";
   GRANT USAGE ON ALL SEQUENCES IN SCHEMA flemingrx TO "<client-id>";
   ```

Creating the tables, loading the demo data, signing in and reading a clinician's patients have
been run against Lakebase from a laptop. Running as a deployed Databricks App has not.

Every time the app starts it brings the database up to date: it creates missing tables, adds
columns the tables have gained, and loads any fabricated clinicians and patients from `seed.py`
that aren't there yet. It only ever adds; nothing a clinician has recorded is changed. The demo
sign-in details are in `seed.py`. To start over, empty the tables and start the app again.

Signing in lasts for the browser session: reloading the page signs you out.

## The bar

How likely an antibiotic must be to work before it is recommended (the "bar") is no longer a fixed
90%. It depends on how sick the patient is, and it is read from the model's own evaluation:

- `policy.severity_level` sorts each patient into Mild, Systemic signs or Severe from their care
  setting and vital signs, using the same cut-offs as notebook 04.
- `data.severity_bars` reads the bar for each level from MLflow: notebook 04 logs them as
  `severity_bar_<level>` on its `recommendation_policy` run, in the same experiment as the
  registered model. The bars are not stored inside the model file.
- If MLflow can't be read (no Databricks sign-in), the app uses notebook 04's defaults and says
  so under the severity line on the assessment.

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

Other settings: `MODEL_NAME` picks a different registered model.

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
