# FlemingRx

**Antibiotic stewardship for urinary tract infections: the narrowest antibiotic that is likely
enough to work, and a platform that follows the patient after the prescription is written.**

Research demo built at a hackathon on de-identified data, with fabricated patients in the app.
Not for clinical use.

## Inspiration

When a patient arrives with a suspected urinary tract infection, the clinician has to pick an
antibiotic before the lab knows which bacteria are there. The culture takes two or three days.
Until then the choice is a guess.

The safe-looking guess is a broad-spectrum drug, because it works against almost everything. But
every broad-spectrum prescription that wasn't needed pushes bacteria toward resistance, and the
drugs of last resort stop working for the patients who really need them. The careful guess is a
narrow drug, which protects those last-resort drugs but fails more often.

Hospitals handle this today with an antibiogram: one table of average resistance rates for the
whole hospital. It gives every patient the same answer. We wanted an answer for *this* patient,
and a tool that stays with the case after the first prescription. The project is named after
Alexander Fleming, who discovered penicillin and warned about resistance in the same breath.

## What it does

FlemingRx is a decision-support tool for choosing an antibiotic for a suspected UTI.

1. **Read.** It reads the patient's record: recent antibiotics, past resistant cultures, earlier
   organisms, recent procedures, and care setting.
2. **Predict.** For each of 7 antibiotics, it estimates the chance the drug will work for *this*
   patient.
3. **Recommend.** It walks the drugs from narrowest to broadest spectrum and suggests the first
   one that clears the bar.

The 7 antibiotics, narrowest first: nitrofurantoin, trimethoprim/sulfamethoxazole, cefazolin,
ciprofloxacin, ceftriaxone, piperacillin/tazobactam, meropenem.

The bar depends on how sick the patient is, because a failed first drug costs a few days in a mild
bladder infection and can be fatal in sepsis:

| Severity | Who | Required chance of working |
|---|---|---|
| Mild | Normal vital signs, outpatient or emergency | 85% |
| Systemic signs | Fever, fast heart rate or breathing, or on a hospital ward | 90% |
| Severe | Intensive care, low blood pressure, or very fast breathing | 95% |

Severity comes from the care setting and the vital signs in the 48 hours before the culture. The
platform works it out and shows the level, the bar it sets, and where that bar came from. The
clinician always makes the call: they accept the recommendation, override it with a reason, or
decide not to treat.

## Digging deeper: the platform

Clinical decision support systems are already used to help clinicians choose an antibiotic.
FlemingRx goes further than the single decision. Its model guides the first prescription, and the
platform around it lets clinicians track each case, reassess it, and revise the treatment when the
culture comes back. By supporting care after the prescription is written, FlemingRx targets the
root of antibiotic overprescription, not one moment in it.

Each clinician signs in and sees only the patients assigned to them.

| Screen | What the clinician does there |
|---|---|
| **Dashboard** | Sees today's counts, the patients who need reassessment, cultures awaiting a result or a review, and a 7-day stewardship snapshot. Everything links to the screen behind it. |
| **Clinical Assessment** | Opens a patient and sees their record, their severity and bar, four pre-checks (symptoms, allergies, kidney function, pregnancy), each antibiotic's predicted chance of working, and the recommendation. Then accepts, overrides with a reason, or decides not to treat. |
| **Culture Results** | Reviews each culture when it comes back: the organism, the lab's susceptibility results, and how the current prescription compares. A resistant result is flagged as a mismatch. |
| **Follow-up** | Works a queue of patients who need another look: a resistance mismatch, symptoms not improving, an adverse reaction, or a finished course. Records the reassessment. |
| **Stewardship Analytics** | Sees totals for a chosen period: which antibiotics were prescribed, how often the recommendation was accepted, resistance by antibiotic, and follow-up outcomes. Figures open the cases behind them. |

Two details we cared about:

- **A positive culture without symptoms usually shouldn't be treated at all.** The pre-checks say
  so before the recommendation, and "do not treat" is a recorded decision, not a missing one.
- **Everything is logged.** Each recommendation is saved with what the model predicted, what the
  pre-checks found and what the clinician decided, so the analytics count real decisions.

## Results

On 15,690 urine cultures from 2022 onward that the models never saw, we compared FlemingRx with the
two one-size-fits-all options: the narrowest drug for everyone, and the broad-spectrum drug the
hospital average points to.

| | Narrowest for all | Broad for all | FlemingRx |
|---|---|---|---|
| Given a broad-spectrum drug | 0% | 100% | 25.8% |
| First drug failed, all patients | 24.83% | 4.44% | 9.07% |
| First drug failed, severe patients | 38.04% | 7.66% | 3.35% |

Only about a quarter of patients get a broad-spectrum drug, and for the sickest patients the first
pick fails less than half as often as broad-for-all.

As a check at equal safety: with a single bar chosen on 2021 data to match the hospital average,
FlemingRx failed for 4.05% of test patients against 4.44%, while recommending a broad-spectrum
drug for 63.9% instead of 100%.

## How we built it

**Data.** The Antibiotic Resistance Microbiology Dataset (ARMD), a public, de-identified dataset
of microbiology cultures from one health system, 2008 onward. We used the urine cultures.

**Pipeline, on Databricks.** Four notebooks, each feeding the next:

| Notebook | What it does |
|---|---|
| `01_ingest` | Loads the raw CSV files into Delta tables. |
| `02_features` | Builds 105 inputs per culture from what was known *before* it was ordered: past resistance, antibiotic exposure, earlier organisms, procedures, care setting, demographics. |
| `03_train` | Trains one gradient-boosting model per antibiotic, compares it against the antibiogram, a prior-resistance rule and logistic regression, and registers the result in Unity Catalog. |
| `04_recommend` | Turns predictions into a recommendation, evaluates it against the one-size-fits-all options, and sets the bar for each severity level. |

**Splitting by time, never shuffling.** Models are trained on 2008 to 2021 and scored once on 2022
onward. Choices such as model settings were made on 2021 alone, so the test years stay honest.

**What the model relies on.** We ranked every input by how much the model's accuracy drops when it
is scrambled. Past resistance matters most by a wide margin, then antibiotic exposure, then
demographics and earlier organisms. The full ranking is in
[docs/model_inputs_ranked.md](docs/model_inputs_ranked.md).

**The app.** A Streamlit app deployed as a Databricks App. It loads the registered model from Unity
Catalog, reads the severity bars from the model's MLflow run, and keeps clinicians, patients,
prescriptions, cultures, follow-ups and the assessment log in Databricks Lakebase (managed
Postgres) through SQLAlchemy. The recommendation rule and the severity rule in the app are tested
against the notebook versions, so the app applies exactly what was evaluated.

## Iteration

We did not stop at the first model that worked.

**More data did not mean a better model.** The core dataset is about 4 GB. ARMD also offers a
separate 19.7 GB file of patients' long-term diagnoses (diabetes, kidney disease and so on), which
takes the data to roughly 20 GB. We built the pipeline to ingest it, turned it into extra inputs
and retrained. The model's average test score (AUROC) moved from 0.736 to 0.737. For five times
the data, that is no meaningful gain, so the final model leaves those inputs out. The code is
still in the repo (`01b_ingest_comorbidity`, switched off by default) so the result can be
re-checked.

**Letting the patient's severity set the bar beat one fixed bar.** Our first version used a single
required chance of working, 90%, chosen by the clinician for every patient. We replaced it with a
bar set from each patient's severity. On the same test cultures, compared with 90% for everyone:

| | 90% bar for everyone | Bar set by severity |
|---|---|---|
| Severe patients: first drug failed | 5.98% | 3.35% |
| Mild patients: given a broad-spectrum drug | 43.2% | 18.2% |
| Mild patients: first drug failed | 5.30% | 9.55% |

The sickest patients get a safer first pick, and mild patients, who are most of the cultures, get
far fewer broad-spectrum drugs. The cost is the last row, and we chose to show it rather than hide
it.

**Other things we tried.** Tuning the gradient-boosting settings, adding "how many days ago"
versions of the history inputs, and adding recent labs and vital signs as model inputs. Each was
measured on 2021 before it was kept or dropped.

## Security

Even with fabricated patients, we built the app the way patient data should be handled.

- **Passwords are never stored.** Each password is kept only as a salted scrypt hash, which cannot
  be turned back into the password, and sign-in checks take the same time whether or not the user
  exists.
- **Patient data is encrypted on the way and at rest.** The app only connects to the database over
  an encrypted (TLS) connection, and the database, Databricks Lakebase, encrypts what it stores.
- **No database password exists.** The app signs in to the database with a token from Databricks
  that expires after an hour, so there is no long-lived secret to leak.
- **Clinicians see only their own patients.** Every lookup is limited to the patients assigned to
  the signed-in clinician, and tests check that one clinician cannot open another's patient,
  culture or follow-up.
- **Least access.** The deployed app has its own identity with access to only the model, the
  tables and the database it needs.

## Challenges we ran into

- **What was known when.** An input must never contain information from after the culture was
  ordered. Getting every "days before" calculation right took more care than the model did.
- **Honest comparisons.** The dataset does not record what doctors actually prescribed, so we
  compare against decision rules, not against real prescribing.
- **A dataset without symptoms.** It cannot tell a bladder infection from a kidney infection in a
  patient with normal vital signs, so "mild" is a rough stand-in.
- **A 20 GB file on free-tier compute.** It was too large to upload directly, so it had to be
  converted on a laptop first.

## What we learned

- Resistance prediction is less about the model and more about careful data: what was known when,
  and what the comparison really means.
- One bar for everyone is the wrong question. The right amount of risk depends on how sick the
  patient is.
- The trade-off is real and we chose to show it: mild patients see more first-drug failures than
  they would on a broad-spectrum drug (9.55% against 4.09%), in exchange for far less
  broad-spectrum use.
- Honest baselines matter. We compare against decision rules, not against what doctors actually
  prescribed, because the dataset does not record that.
- More data is not automatically better. Five times the data bought us almost nothing.

## What's next for FlemingRx

- Validate on data from other hospitals, where resistance patterns differ.
- Sharpen severity: "mild" is a rough stand-in, since the dataset has no symptoms.
- Make the checks a real prescription needs clinically sound. The app's pre-checks for allergies,
  kidney function and pregnancy are a few simple rules written for the demo; they need review by
  clinicians and pharmacists, and drug interactions are not checked at all.
- Let the clinician adjust the severity level and the dose, and record why.
- Test it inside a live clinic workflow, connected to the health record instead of fabricated
  patients. That is the gap the research identified.

## Limits

- **Decision support, not a prescription.** A clinician makes the choice.
- **One health system.** Other hospitals will have different resistance patterns.
- **Seven antibiotics, urine cultures only.**
- **The app's patients are fabricated.** So are their prescriptions, cultures and follow-ups.
- **Scored on fully tested cultures.** Only cultures where the lab tested all 7 antibiotics were
  scored. Those left out tend to grow more resistant organisms, so real failure rates may be
  higher.

## In this repository

| Folder | What's in it |
|---|---|
| `notebooks/` | The four pipeline notebooks (and the optional `01b`), written for Databricks. |
| `app/` | The Streamlit app. [app/README.md](app/README.md) covers how it is built, the database, and how to run and deploy it. |
| `tests/` | 40 tests: the recommendation and severity rules against the notebooks, the database, sign-in and access, and the analytics counts. |
| `docs/` | The ranking of the model's inputs. |
| `local/` | Helpers for running a notebook on a laptop and converting the large diagnoses file. |

To run the app on a laptop:

```bash
pip install -r app/requirements.txt
```

```bash
cd app && streamlit run app.py
```

It needs the Databricks CLI signed in for the database, or `DATABASE_URL=sqlite:///flemingrx.db`
to use a local file instead. Details are in [app/README.md](app/README.md).

To run the tests:

```bash
pip install pytest && pytest tests
```
