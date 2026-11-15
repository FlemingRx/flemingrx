"""FlemingRx demo app: patient details in, a recommended antibiotic out.

Run locally with `streamlit run app.py` from this folder (see README.md for the two settings a
local run needs). On Databricks it runs as a Databricks App using app.yaml.

The look lives in three places: .streamlit/config.toml (brand colours and fonts), styles.css
(cards, header, the result panel) and assets/ (the logo).
"""

import base64
import html
import os
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

import data
import policy

HERE = pathlib.Path(__file__).parent
LOGO = HERE / "assets" / "flemingrx-logo.png"
MARK = HERE / "assets" / "flemingrx-mark.png"

NAVY, TEAL = "#132557", "#009788"

st.set_page_config(page_title="FlemingRx · UTI antibiotic stewardship", page_icon=str(MARK),
                   layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    model = data.load_model()
    return model, data.model_features(model)


@st.cache_data(show_spinner="Loading the evaluation results...")
def get_table(name: str) -> pd.DataFrame:
    return data.load_table(name)


@st.cache_data
def image_data_uri(path: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ----------------------------------------------------------------------------- building blocks

def card(key: str, icon: str, title: str, hint: str | None = None):
    """A white card with a header row, like the sections of a clinical form. Returns its body."""
    box = st.container(border=True, key=f"card_{key}")
    with box.container(key=f"cardhead_{key}", horizontal=True, vertical_alignment="center"):
        st.markdown(f":material/{icon}: **{title}**")
        if hint:
            st.caption(hint)
    return box


def top_bar(source: str):
    st.html(f"""
    <div class="fx-topbar">
      <div class="fx-brand">
        <img src="{image_data_uri(LOGO)}" alt="FlemingRx" />
        <span class="fx-divider"></span>
        <span class="fx-product">UTI Antibiotic Stewardship</span>
      </div>
      <div class="fx-topbar-right">
        <span class="fx-chip fx-chip-teal"><span class="fx-dot"></span>Stewardship mode</span>
        <span class="fx-chip" title="{html.escape(source)}">Research demo</span>
      </div>
    </div>""")


SPECTRUM_CLASS = {"Narrow": "narrow", "Medium": "medium", "Broad": "broad"}


# ----------------------------------------------------------------------------- patient form

def patient_form(features) -> dict:
    """The short form. Returns the patient as {feature name: number}, the model's input."""
    patient = data.blank_patient(features)

    with card("demo", "person", "Demographics"):
        age = st.select_slider("Age", list(data.AGE_BANDS), value="65-74", key="age")
        sex = st.segmented_control("Biological sex", list(data.SEX_CODES), default="Female",
                                   required=True, key="sex", width="stretch")
    patient["age_min"] = float(data.AGE_BANDS[age])
    patient["gender"] = float(data.SEX_CODES[sex])

    with card("setting", "local_hospital", "Care setting"):
        setting = st.pills("Where the culture was ordered", list(data.SETTINGS),
                           default="Outpatient clinic", required=True, key="setting")
        nursing_home = st.toggle("Nursing home contact on record", key="nursing_home")
    patient.update({name: float(value) for name, value in data.SETTINGS[setting].items()})
    patient["nursing_home"] = float(nursing_home)

    resistance = {policy.drug_label(name[len("prior_resist_"):]): name
                  for name in features if name.startswith("prior_resist_")}
    with card("resist", "coronavirus", "Resistance history", "past 12 months"):
        past_resistance = st.pills(
            "Earlier culture resistant to", list(resistance), selection_mode="multi",
            key="past_resistance",
            help="An earlier culture, at least 3 days old, that was resistant to this antibiotic.",
        )
        other_resistance = st.number_input(
            "Other antibiotics with resistance (how many)", 0, 25, 0, key="other_resistance",
            help="Antibiotics outside the 7 above that an earlier culture was resistant to.",
        )
    for label in past_resistance:
        patient[resistance[label]] = 1.0
    patient["n_prior_resistant_drugs"] = float(len(past_resistance) + other_resistance)

    classes = data.options(features, "abx_class_", "_90d")
    with card("abx", "medication", "Antibiotic exposure"):
        recent_classes = st.multiselect("Classes taken in the last 90 days", list(classes),
                                        placeholder="None selected", key="recent_classes")
        last_options = {"Last month": 14, "1–3 months ago": 60}
        if not recent_classes:
            last_options.update({"3–12 months ago": 180, "None in past year": data.NO_RECENT_ANTIBIOTIC})
        last = st.pills("Most recent antibiotic of any kind", list(last_options),
                        default=list(last_options)[-1], required=True,
                        key=f"last_abx_{len(last_options)}")
    for label in recent_classes:
        patient[classes[label]] = 1.0
    patient["days_since_last_antibiotic"] = float(last_options[last])

    organisms = data.options(features, "prior_org_")
    procedures = data.options(features, "proc_")
    with card("micro", "biotech", "Microbiology & procedures"):
        for label in st.multiselect("Organisms grown in the past year", list(organisms),
                                    placeholder="None selected", key="organisms"):
            patient[organisms[label]] = 1.0
        for label in st.pills("Procedures in the last 90 days", list(procedures),
                              selection_mode="multi", key="procedures"):
            patient[procedures[label]] = 1.0

    return patient


# ----------------------------------------------------------------------------- result panel

def hero_html(drug: str, chance: float, spectrum: str, clears: bool, target: float, why: str) -> str:
    pct = round(100 * chance)
    status = (f'<span class="fx-tag fx-tag-ok">✓ Clears the {target:.0%} bar</span>' if clears else
              f'<span class="fx-tag fx-tag-warn">! Nothing reaches {target:.0%}: broadest offered</span>')
    return f"""
    <div class="fx-hero">
      <div class="fx-hero-text">
        <div class="fx-eyebrow">Recommended empiric antibiotic</div>
        <div class="fx-drug">{html.escape(drug)}</div>
        <div class="fx-tags">
          <span class="fx-tag fx-spec-{SPECTRUM_CLASS[spectrum]}">{spectrum} spectrum</span>
          {status}
        </div>
        <p class="fx-why">{html.escape(why)}</p>
      </div>
      <div class="fx-gauge" style="background: conic-gradient(#2BD4BD {pct * 3.6}deg, rgba(255,255,255,.14) 0deg);">
        <div class="fx-gauge-inner"><b>{pct}%</b><span>chance of<br>working</span></div>
      </div>
    </div>"""


def ladder_html(drugs, risks, picked: int, target: float) -> str:
    rows = []
    for position, (drug, risk) in enumerate(zip(drugs, risks)):
        chance = 1 - risk
        spectrum = policy.spectrum(drug)
        state = "picked" if position == picked else ("pass" if risk <= 1 - target else "fail")
        badge = '<span class="fx-pick">Recommended</span>' if position == picked else ""
        rows.append(f"""
        <div class="fx-row fx-row-{state}">
          <div class="fx-rank">{position + 1}</div>
          <div class="fx-name">{html.escape(policy.drug_label(drug))}
            <span class="fx-spec fx-spec-{SPECTRUM_CLASS[spectrum]}">{spectrum}</span>{badge}</div>
          <div class="fx-track">
            <div class="fx-fill" style="width:{100 * chance:.1f}%"></div>
            <div class="fx-mark" style="left:{100 * target:.1f}%"></div>
          </div>
          <div class="fx-pct">{100 * chance:.0f}%</div>
        </div>""")
    return f"""
    <div class="fx-ladder">
      <div class="fx-ladder-axis"><span>Narrowest</span><span class="fx-arrow"></span><span>Broadest</span></div>
      {''.join(rows)}
      <div class="fx-legend">
        <span><i class="fx-key fx-key-pass"></i>Clears the bar</span>
        <span><i class="fx-key fx-key-fail"></i>Below the bar</span>
        <span><i class="fx-key fx-key-mark"></i>Your bar ({target:.0%})</span>
      </div>
    </div>"""


def recommendation_panel(model, features, patient):
    predictions = data.predict(model, features, patient)     # chance of resistance, narrowest first
    drugs = list(predictions.index)
    risks = list(predictions.to_numpy())

    with st.container(key="results"):
        with st.container(border=True, key="card_target"):
            target = st.slider(
                "Required chance of working", 80, 99,
                int(round(100 * float(os.getenv("TARGET_SUCCESS", "0.90")))), format="%d%%",
                key="target_pct",
                help="The rule recommends the narrowest antibiotic that is at least this likely "
                     "to work. Higher is safer for this patient but leads to more broad-spectrum use.",
            ) / 100

        picked = policy.recommend(risks, target)
        st.html(hero_html(policy.drug_label(drugs[picked]), 1 - risks[picked],
                          policy.spectrum(drugs[picked]), risks[picked] <= 1 - target, target,
                          policy.explain(risks, drugs, target)))

        with card("ladder", "stairs", "Every option, narrowest first", "predicted chance of working"):
            st.html(ladder_html(drugs, risks, picked, target))
    return target


# ----------------------------------------------------------------------------- other tabs

def evidence_tab(target: float):
    try:
        tradeoff = get_table("gold_policy_tradeoff")
    except Exception as error:   # shown to the user: the app still works without this table
        st.info(f"The evaluation results aren't available here, so this page is empty. ({error})",
                icon=":material/info:")
        return
    tradeoff = tradeoff.replace({"policy": {"model": "Model (FlemingRx)",
                                            "antibiogram": "Antibiogram (today)"}})
    model_rows = tradeoff[tradeoff["policy"] == "Model (FlemingRx)"]
    here = model_rows.iloc[(model_rows["target_success"] - target).abs().argsort()[:1]]

    chart_col, side_col = st.columns([7, 3], gap="medium")
    with chart_col, card("chart", "show_chart", "How the rule performed on cultures the model never saw"):
        # The model gives a smooth curve. The antibiogram gives everyone the same drug, so it only
        # has a few distinct outcomes: show those as separate dots rather than joining them up.
        colors = alt.Scale(domain=["Model (FlemingRx)", "Antibiogram (today)"], range=[TEAL, NAVY])
        base = alt.Chart(tradeoff).encode(
            x=alt.X("broad_pct", title="Patients given a broad-spectrum antibiotic (%)"),
            y=alt.Y("failure_pct", title="Recommended antibiotic failed (%)"),
            color=alt.Color("policy", title=None, scale=colors,
                            legend=alt.Legend(orient="top", labelFontSize=12, symbolSize=120)),
            tooltip=[alt.Tooltip("policy", title="Policy"),
                     alt.Tooltip("target_success", title="Bar", format=".1%"),
                     alt.Tooltip("failure_pct", title="Failed (%)"),
                     alt.Tooltip("broad_pct", title="Broad (%)"),
                     alt.Tooltip("narrow_pct", title="Narrow (%)")],
        )
        is_model = alt.datum.policy == "Model (FlemingRx)"
        you = alt.Chart(here).mark_point(size=420, shape="circle", filled=False, strokeWidth=2.5,
                                         color=NAVY).encode(x="broad_pct", y="failure_pct")
        chart = (base.transform_filter(is_model).mark_line(point=alt.OverlayMarkDef(size=28),
                                                             strokeWidth=2.5)
                 + base.transform_filter(~is_model).mark_point(size=170, filled=True, opacity=0.9)
                 + you).properties(height=380)
        chart = chart.configure_axis(gridColor="#EDF1F6", domainColor="#D5DCE6", tickColor="#D5DCE6",
                                     labelColor="#5B6782", titleColor="#3A4560", titleFontWeight=600,
                                     labelFont="Inter", titleFont="Inter").configure_view(strokeWidth=0).configure(background="#FFFFFF")
        st.altair_chart(chart, width="stretch")
        st.caption("Each point is one setting of the rule on the 2022+ test cultures. Down and to the "
                   "left is better: fewer failures and less broad-spectrum use. The antibiogram gives "
                   "every patient the same drug, so it has only a few possible outcomes. From notebook 04.")
    with side_col:
        row = here.iloc[0]
        with card("atbar", "my_location", "At your bar"):
            st.caption(f"The ringed point: the model with a {row['target_success']:.1%} bar, "
                       f"set on the Recommend tab.")
            st.metric("Recommended antibiotic failed", f"{row['failure_pct']:.1f}%")
            st.metric("Given a broad-spectrum antibiotic", f"{row['broad_pct']:.1f}%")
            st.metric("Given a narrow-spectrum antibiotic", f"{row['narrow_pct']:.1f}%")


LIMITS = [
    ("stethoscope", "Decision support, not a prescription",
     "A clinician makes the choice. The tool shows how likely each option is to work."),
    ("rule", "It only predicts resistance",
     "It does not check allergies, pregnancy, kidney function, drug interactions, or the site and "
     "severity of the infection."),
    ("location_city", "7 antibiotics, one health system",
     "Trained on positive urine cultures from one health system, 2008 onwards. Other hospitals "
     "will have different resistance patterns."),
    ("filter_alt", "Scored on fully tested cultures",
     "Only cultures where the lab tested all 7 antibiotics were scored. Those left out tend to grow "
     "more resistant organisms, so real failure rates may be higher (notebook 04, section 6)."),
    ("keyboard", "Details are typed in by hand",
     "Reading the patient from medical records is planned, not built."),
]


def limits_tab():
    columns = st.columns(2, gap="medium")
    for index, (icon, title, body) in enumerate(LIMITS):
        with columns[index % 2], card(f"limit{index}", icon, title):
            st.write(body)


# ----------------------------------------------------------------------------- page

st.html(HERE / "styles.css")
top_bar(data.model_source())

try:
    model, features = get_model()
except Exception as error:   # shown to the user: nothing works without the model
    st.error(f"The model could not be loaded from {data.model_source()}. {error}",
             icon=":material/error:")
    st.caption("Running locally? Install the requirements first: pip install -r app/requirements.txt")
    st.stop()

recommend_tab, evidence, limits = st.tabs([":material/clinical_notes: Recommend",
                                           ":material/insights: Evidence",
                                           ":material/info: Limits"])
with recommend_tab:
    form_column, result_column = st.columns([5, 7], gap="large")
    with form_column:
        st.html('<div class="fx-formhead"><div class="fx-title">Patient details</div>'
                '<div class="fx-sub">Enter risk factors. The estimate updates as you go.</div></div>')
        patient = patient_form(features)
    with result_column:
        target = recommendation_panel(model, features, patient)
with evidence:
    evidence_tab(target)
with limits:
    limits_tab()

st.html(f"""<div class="fx-footer">
  <img src="{image_data_uri(MARK)}" alt="" />
  <span>Research demo built on de-identified data. <b>Not for clinical use.</b></span>
  <span class="fx-footer-src">Model: {html.escape(data.model_source())}</span>
</div>""")
