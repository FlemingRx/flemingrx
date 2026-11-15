"""FlemingRx: an antibiotic stewardship platform for clinicians.

This file is the frame around every screen (sign-in, sidebar, top bar, footer) and holds the
clinical assessment: the signed-in clinician's patients, and for each one a recommended antibiotic.
The dashboard is in dashboard.py, the database in db.py. Screens that are planned but not built
yet are listed in PLANNED below.

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

import dashboard
import data
import db
import policy

HERE = pathlib.Path(__file__).parent
LOGO = HERE / "assets" / "flemingrx-logo.png"
LOGO_LIGHT = HERE / "assets" / "flemingrx-logo-light.png"   # white wordmark, for the navy sidebar
MARK = HERE / "assets" / "flemingrx-mark.png"

NAVY, TEAL = "#132557", "#009788"

st.set_page_config(page_title="FlemingRx · Antibiotic stewardship", page_icon=str(MARK),
                   layout="wide", initial_sidebar_state="auto")


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    model = data.load_model()
    return model, data.model_features(model)


@st.cache_data(show_spinner="Loading the evaluation results...")
def get_table(name: str) -> pd.DataFrame:
    return data.load_table(name)


@st.cache_resource(show_spinner="Connecting to the database...")
def database_ready() -> bool:
    db.init()
    return True


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


def top_bar(page_title: str, sample_data: bool):
    sample = '<span class="fx-chip fx-chip-amber">Sample data</span>' if sample_data else ""
    st.html(f"""
    <div class="fx-topbar">
      <div class="fx-crumbs">
        <span>FlemingRx</span><span class="fx-crumb-sep">›</span><b>{html.escape(page_title)}</b>
      </div>
      <div class="fx-topbar-right">
        <span class="fx-chip fx-chip-teal"><span class="fx-dot"></span>Stewardship mode</span>
        {sample}
        <span class="fx-chip" title="{html.escape(data.model_source())}">Research demo</span>
      </div>
    </div>""")


SPECTRUM_CLASS = {"Narrow": "narrow", "Medium": "medium", "Broad": "broad"}


# ----------------------------------------------------------------------------- patients

def last_antibiotic_text(days: int) -> str:
    if days >= data.NO_RECENT_ANTIBIOTIC:
        return "None in the past year"
    return f"About {days} days ago"


def names(slugs, label=data.pretty) -> str:
    return ", ".join(label(slug) for slug in slugs) or "None"


def patient_summary(record: dict) -> str:
    """The one-line description under a patient number."""
    parts = [record["age_band"], record["sex"], record["care_setting"]]
    if record["resistant_to"]:
        parts.append(f"{len(record['resistant_to'])} prior resistance")
    return " · ".join(parts)


def patient_list(clinician: dict):
    records = db.patients_for(clinician["id"])
    st.html(f'<div class="fx-formhead"><div class="fx-title">My patients</div>'
            f'<div class="fx-sub">{len(records)} assigned to {html.escape(clinician["name"])}. '
            f'Select a patient to open their assessment.</div></div>')
    if not records:
        st.info("No patients are assigned to you yet.", icon=":material/info:")
        return
    with st.container(key="patientlist"):
        for record in records:
            number = record["patient_number"]
            if st.button(f"**{number}** :gray[{patient_summary(record)}]", key=f"open_{number}",
                         icon=":material/chevron_right:", width="stretch"):
                st.session_state["patient"] = number
                st.rerun()


def record_card(key: str, icon: str, title: str, rows: list, hint: str | None = None):
    """A read-only card of (label, value) rows from the patient's record."""
    body = "".join(f'<div class="fx-kv"><span>{html.escape(label)}</span>'
                   f'<b>{html.escape(str(value))}</b></div>' for label, value in rows)
    with card(key, icon, title, hint):
        st.html(body)


def patient_record(record: dict):
    record_card("demo", "person", "Demographics",
                [("Age", record["age_band"]), ("Biological sex", record["sex"])])
    record_card("setting", "local_hospital", "Care setting",
                [("Where the culture was ordered", record["care_setting"]),
                 ("Nursing home contact on record", "Yes" if record["nursing_home"] else "No")])
    record_card("resist", "coronavirus", "Resistance history",
                [("Earlier culture resistant to", names(record["resistant_to"], policy.drug_label)),
                 ("Other antibiotics with resistance", record["other_resistant_count"])],
                "past 12 months")
    record_card("abx", "medication", "Antibiotic exposure",
                [("Classes taken in the last 90 days", names(record["recent_antibiotic_classes"])),
                 ("Most recent antibiotic of any kind",
                  last_antibiotic_text(record["days_since_last_antibiotic"]))])
    record_card("micro", "biotech", "Microbiology & procedures",
                [("Organisms grown in the past year", names(record["prior_organisms"])),
                 ("Procedures in the last 90 days", names(record["recent_procedures"]))])


# ----------------------------------------------------------------------------- result panel

def hero_html(drug: str, chance: float, spectrum: str, clears: bool, target: float, why: str) -> str:
    pct = round(100 * chance)
    status = (f'<span class="fx-tag fx-tag-ok">✓ Clears the {target:.0%} bar</span>' if clears else
              f'<span class="fx-tag fx-tag-warn">! Nothing reaches {target:.0%}: broadest offered</span>')
    return f"""
    <div class="fx-hero">
      <div class="fx-hero-text">
        <div class="fx-eyebrow">Recommended empiric antibiotic</div>
        <div class="fx-drug">{html.escape(drug).replace("/", "/<wbr>")}</div>
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
    ("database", "Patients are fabricated",
     "Patient details come from the app's own database of made-up demo patients. Reading them "
     "from a hospital's records is planned, not built."),
]


def limits_tab():
    columns = st.columns(2, gap="medium")
    for index, (icon, title, body) in enumerate(LIMITS):
        with columns[index % 2], card(f"limit{index}", icon, title):
            st.write(body)


# ----------------------------------------------------------------------------- screens

def assessment_page():
    clinician = st.session_state["clinician"]
    number = st.session_state.get("patient")
    # patient_for only returns patients assigned to this clinician, so nobody else's can be opened.
    record = db.patient_for(clinician["id"], number) if number else None
    if record is None:
        patient_list(clinician)
    else:
        patient_assessment(record)


def patient_assessment(record: dict):
    if st.button("All patients", key="back", icon=":material/arrow_back:", type="tertiary"):
        del st.session_state["patient"]
        st.rerun()
    try:
        model, features = get_model()
    except Exception as error:   # shown to the user: nothing works without the model
        st.error(f"The model could not be loaded from {data.model_source()}. {error}",
                 icon=":material/error:")
        st.caption("Running locally? Install the requirements first: pip install -r app/requirements.txt")
        return

    recommend_tab, evidence, limits = st.tabs([":material/clinical_notes: Recommend",
                                               ":material/insights: Evidence",
                                               ":material/info: Limits"])
    with recommend_tab:
        record_column, result_column = st.columns([5, 7], gap="large")
        with record_column:
            st.html(f'<div class="fx-formhead"><div class="fx-title">'
                    f'{html.escape(record["patient_number"])}</div>'
                    f'<div class="fx-sub">Urinary tract infection. Details from the patient record.'
                    f'</div></div>')
            patient_record(record)
        with result_column:
            target = recommendation_panel(model, features, data.patient_features(features, record))
    with evidence:
        evidence_tab(target)
    with limits:
        limits_tab()


def sign_in_page():
    _, middle, _ = st.columns([1, 1, 1])
    with middle, st.container(border=True, key="card_signin"):
        st.html(f'<div class="fx-signin-head"><img src="{image_data_uri(LOGO)}" alt="FlemingRx" />'
                f'<div class="fx-title">Log in</div></div>')
        with st.form("signin", border=False):
            username = st.text_input("User ID", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Log in", type="primary", width="stretch")
        if submitted:
            clinician = db.sign_in(username, password)
            if clinician:
                st.session_state["clinician"] = clinician
                st.rerun()
            st.error("Incorrect credentials", icon=":material/lock:")


# Screens from the roadmap that are not built yet: (title, address, icon, what it will hold).
PLANNED = [
    ("Patient Encounters", "patients", "group",
     "Each patient's wider history in one place: allergies, current medicines, kidney and liver "
     "function, earlier infections and past culture results."),
    ("Culture Results", "cultures", "biotech",
     "Urine culture and susceptibility results as they come back, with a flag when the empiric "
     "choice should be reconsidered or can be narrowed."),
    ("Follow-up", "follow-up", "event_repeat",
     "What happens after treatment starts: response, side effects, reassessment reminders and "
     "treatment completion."),
    ("Stewardship Analytics", "analytics", "bar_chart",
     "Prescribing across the service: broad versus narrow use, recommendations accepted or "
     "overridden, and resistance trends."),
]


def planned_page(icon: str, title: str, body: str):
    def page():
        st.html(f"""<div class="fx-empty">
          <span class="fx-badge fx-badge-navy">Planned</span>
          <div class="fx-title">{html.escape(title)}</div>
          <p>{html.escape(body)}</p>
        </div>""")
    return page


# ----------------------------------------------------------------------------- frame

st.html(HERE / "styles.css")
try:
    database_ready()
except Exception as error:   # shown to the user: nobody can sign in without the database
    st.error(f"The database ({db.database_kind()}) could not be reached. {error}",
             icon=":material/error:")
    st.stop()

if "clinician" not in st.session_state:
    st.navigation([st.Page(sign_in_page, title="Log in")], position="hidden").run()
    st.stop()

clinician = st.session_state["clinician"]
assessment = st.Page(assessment_page, title="Clinical Assessment", url_path="assessment",
                     icon=":material/clinical_notes:")
home = st.Page(lambda: dashboard.render(clinician), title="Dashboard", url_path="dashboard",
               icon=":material/dashboard:", default=True)
planned = [st.Page(planned_page(icon, title, body), title=title, url_path=path,
                   icon=f":material/{icon}:") for title, path, icon, body in PLANNED]
current = st.navigation({"Workspace": [home, assessment], "Coming next": planned})
if current is not assessment:      # coming back to the assessment starts at the patient list
    st.session_state.pop("patient", None)

st.logo(str(LOGO_LIGHT), size="large", icon_image=str(MARK))
with st.sidebar:
    st.html(f'<div class="fx-sidenote"><b>Signed in</b>{html.escape(clinician["name"])}<br>'
            f'{html.escape(clinician["role"])}</div>')
    if st.button("Log out", key="signout", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.html('<div class="fx-sidenote"><b>Research demo</b>Fabricated patients. '
            'Not for clinical use.</div>')
top_bar(current.title, sample_data=current in (home, assessment))

current.run()

st.html(f"""<div class="fx-footer">
  <img src="{image_data_uri(MARK)}" alt="" />
  <span>Research demo with fabricated patients. <b>Not for clinical use.</b></span>
  <span class="fx-footer-src">Model: {html.escape(data.model_source())} · Database: {db.database_kind()}</span>
</div>""")
