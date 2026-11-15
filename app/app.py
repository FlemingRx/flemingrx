"""FlemingRx demo app: patient details in, a recommended antibiotic out.

Run locally with `streamlit run app.py` from this folder (see README.md for the two settings a
local run needs). On Databricks it runs as a Databricks App using app.yaml.
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

import data
import policy

st.set_page_config(page_title="FlemingRx", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    model = data.load_model()
    return model, data.model_features(model)


@st.cache_data(show_spinner="Loading the evaluation results...")
def get_table(name: str) -> pd.DataFrame:
    return data.load_table(name)


# ----------------------------------------------------------------------------- patient form

def patient_form(features) -> dict:
    """The short form. Returns the patient as {feature name: number}, the model's input."""
    patient = data.blank_patient(features)

    st.subheader("Patient")
    left, right = st.columns(2)
    age = left.selectbox("Age", list(data.AGE_BANDS), index=5)
    sex = right.radio("Sex", list(data.SEX_CODES), horizontal=True)
    setting = left.selectbox("Where the culture was ordered", list(data.SETTINGS))
    right.write("")
    nursing_home = right.checkbox("Nursing home contact on record")
    patient["age_min"] = float(data.AGE_BANDS[age])
    patient["gender"] = float(data.SEX_CODES[sex])
    patient.update({name: float(value) for name, value in data.SETTINGS[setting].items()})
    patient["nursing_home"] = float(nursing_home)

    st.subheader("History")
    resistance = {policy.drug_label(name[len("prior_resist_"):]): name
                  for name in features if name.startswith("prior_resist_")}
    past_resistance = st.multiselect(
        "Resistant infection in the past year to", list(resistance),
        help="An earlier culture, at least 3 days old, that was resistant to this antibiotic.",
    )
    other_resistance = st.number_input(
        "Other antibiotics with resistance in the past year (how many)", 0, 25, 0,
        help="Antibiotics outside the 7 above that an earlier culture was resistant to.",
    )
    for label in past_resistance:
        patient[resistance[label]] = 1.0
    patient["n_prior_resistant_drugs"] = float(len(past_resistance) + other_resistance)

    classes = data.options(features, "abx_class_", "_90d")
    recent_classes = st.multiselect("Antibiotics taken in the last 90 days (by class)", list(classes))
    for label in recent_classes:
        patient[classes[label]] = 1.0
    last_options = {"In the last month": 14, "1 to 3 months ago": 60}
    if not recent_classes:
        last_options.update({"3 to 12 months ago": 180,
                             "None in the past year": data.NO_RECENT_ANTIBIOTIC})
    last = st.selectbox("Most recent antibiotic of any kind", list(last_options),
                        index=len(last_options) - 1)
    patient["days_since_last_antibiotic"] = float(last_options[last])

    organisms = data.options(features, "prior_org_")
    for label in st.multiselect("Organisms found in cultures in the past year", list(organisms)):
        patient[organisms[label]] = 1.0

    procedures = data.options(features, "proc_")
    for label in st.multiselect("Procedures in the last 90 days", list(procedures)):
        patient[procedures[label]] = 1.0

    return patient


# ----------------------------------------------------------------------------- screens

def recommendation_panel(model, features, patient, target):
    predictions = data.predict(model, features, patient)     # chance of resistance, narrowest first
    drugs = list(predictions.index)
    risks = list(predictions.to_numpy())
    picked = policy.recommend(risks, target)

    st.subheader("Recommendation")
    st.metric("Recommended antibiotic", policy.drug_label(drugs[picked]))
    st.write(policy.explain(risks, drugs, target))

    table = pd.DataFrame({
        "Antibiotic": [policy.drug_label(d) for d in drugs],
        "Spectrum": [policy.spectrum(d) for d in drugs],
        "Chance of working": [100 * (1 - r) for r in risks],
        "Clears the bar": [r <= 1 - target for r in risks],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={"Chance of working": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100)},
    )
    st.caption(f"Antibiotics are listed from narrowest to broadest spectrum. The bar is a "
               f"{target:.0%} chance of working; change it in the sidebar.")


def evidence_tab():
    st.subheader("How the rule performed on cultures the model never saw")
    try:
        tradeoff = get_table("gold_policy_tradeoff")
    except Exception as error:   # shown to the user: the app still works without this table
        st.info(f"The evaluation results aren't available here, so this page is empty. ({error})")
        return
    tradeoff = tradeoff.replace({"policy": {"model": "Model (ours)", "antibiogram": "Antibiogram (today)"}})
    # The model gives a smooth curve. The antibiogram gives everyone the same drug, so it only has a
    # few distinct outcomes: show those as separate dots rather than joining them with a line.
    base = alt.Chart(tradeoff).encode(
        x=alt.X("broad_pct", title="Patients given a broad-spectrum antibiotic (%)"),
        y=alt.Y("failure_pct", title="Recommended antibiotic failed (%)"),
        color=alt.Color("policy", title=None),
        tooltip=["policy", "target_success", "failure_pct", "broad_pct", "narrow_pct"],
    )
    is_model = alt.datum.policy == "Model (ours)"
    chart = (base.transform_filter(is_model).mark_line(point=True)
             + base.transform_filter(~is_model).mark_point(size=160, filled=True))
    st.altair_chart(chart, width="stretch")
    st.caption("Each point is one setting of the rule, on the 2022+ test cultures. Down and to the "
               "left is better: fewer failures and less broad-spectrum use. The antibiogram gives "
               "every patient the same drug, so it has only a few possible outcomes. From notebook 04.")


def limits_tab():
    st.subheader("What this tool does not do")
    st.markdown(
        "- **It is decision support, not a prescription.** A clinician makes the choice.\n"
        "- **It only predicts resistance.** It does not check allergies, pregnancy, kidney function, "
        "drug interactions or the site and severity of the infection.\n"
        "- **It covers 7 antibiotics for positive urine cultures** from one health system, 2008 "
        "onwards. Other hospitals will have different resistance patterns.\n"
        "- **It was scored only on cultures where the lab tested all 7 antibiotics.** The cultures "
        "left out tend to grow more resistant organisms, so real failure rates may be higher "
        "(notebook 04, section 6).\n"
        "- **Patient details are typed in by hand here.** Reading them from medical records is "
        "planned, not built."
    )


# ----------------------------------------------------------------------------- page

st.title("FlemingRx")
st.caption("Which antibiotic is likely to work for this urine infection, before the lab result is back?")

try:
    model, features = get_model()
except Exception as error:   # shown to the user: nothing works without the model
    st.error(f"The model could not be loaded from {data.model_source()}. {error}")
    st.caption("Running locally? Install the requirements first: pip install -r app/requirements.txt")
    st.stop()

with st.sidebar:
    st.header("Setting")
    target = st.slider(
        "Required chance of working (%)", 80, 99,
        int(round(100 * float(os.getenv("TARGET_SUCCESS", "0.90")))),
        help="The rule recommends the narrowest antibiotic that is at least this likely to work.",
    ) / 100
    st.caption("Higher is safer for this patient but leads to more broad-spectrum use.")
    st.divider()
    st.caption(f"Model: {data.model_source()}")

recommend_tab, evidence, limits = st.tabs(["Recommendation", "Evidence", "Limits"])
with recommend_tab:
    form_column, result_column = st.columns([5, 6], gap="large")
    with form_column:
        patient = patient_form(features)
    with result_column:
        recommendation_panel(model, features, patient, target)
with evidence:
    evidence_tab()
with limits:
    limits_tab()

st.divider()
st.caption("Research demo built on de-identified data. Not for clinical use.")
