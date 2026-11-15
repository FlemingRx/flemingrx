"""FlemingRx: an antibiotic stewardship platform for clinicians.

This file is the frame around every screen (sign-in, sidebar, top bar, footer) and holds the
clinical assessment: the signed-in clinician's patients, and for each one a recommended antibiotic.
The dashboard is in dashboard.py, culture results in culture_results.py, follow-up in follow_up.py, analytics in analytics.py, the database in db.py.

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

import analytics
import culture_results
import dashboard
import follow_up
import data
import db
import policy
import prechecks
from ui import card, patient_record

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
    search = st.text_input("Search patient ID", placeholder="Search patient ID...",
                           icon=":material/search:", label_visibility="collapsed",
                           key="patient_search", width=280)
    shown = [r for r in records if search.strip().lower() in r["patient_number"].lower()]
    only = st.session_state.get("patient_filter")     # set when opened from the analytics page
    if only:
        shown = [r for r in shown if r["patient_number"] in only["numbers"]]
        with st.container(horizontal=True, vertical_alignment="center", key="patient_filter_bar"):
            st.markdown(f":material/filter_alt: Showing: **{only['label']}**")
            if st.button("Show all patients", key="patient_filter_clear", type="tertiary"):
                del st.session_state["patient_filter"]
                st.rerun()
    if not shown:
        st.caption("No patients match.")
    with st.container(key="patientlist"):
        for record in shown:
            number = record["patient_number"]
            if st.button(f"**{number}** :gray[{patient_summary(record)}]", key=f"open_{number}",
                         icon=":material/chevron_right:", width="stretch"):
                st.session_state["patient"] = number
                st.rerun()


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


CHECK_WORD = {"ok": "Clear", "unknown": "Not on record", "caution": "Check", "stop": "Stop and review"}


def precheck_strip(record: dict, recommended: str):
    """The four checks to make before acting on the recommendation, in a row above it."""
    checks = prechecks.run(record, recommended)
    cells = "".join(
        f'<div class="fx-check fx-check-{check["level"]}">'
        f'<span class="fx-eyebrow-dark">{check["name"]}</span>'
        f'<b>{html.escape(check["value"])}</b>'
        f'<i>{CHECK_WORD[check["level"]]}</i>'
        f'<small>{html.escape(check["note"])}</small></div>' for check in checks)
    st.html(f'<div class="fx-checks">{cells}</div>')
    if checks[0]["level"] == "stop":      # the symptoms check: treatment may not be needed at all
        st.html('<div class="fx-verdict fx-verdict-bad"><b>Treatment is usually not indicated</b>'
                '<span>No urinary symptoms are documented. The recommendation below shows what the '
                'model would choose if you decide to treat.</span></div>')


def regimen_text(drug: str) -> str:
    dose, frequency, days = policy.USUAL_REGIMENS[drug]
    return f"{dose} · {frequency} · {days} days"


def decision_card(clinician: dict, record: dict, drugs: list, recommended: str):
    """Accept or override the recommendation, or show the prescription already on record."""
    number = record["patient_number"]
    course = db.prescription_for(clinician["id"], number)
    with card("decision", "prescriptions", "Prescription"):
        if course:
            accepted = course["drug"] == course["recommended_drug"]
            decision = ("Accepted the recommendation" if accepted else
                        f"Overrode the recommendation ({policy.drug_label(course['recommended_drug'])})")
            rows = [("Prescribed", policy.drug_label(course["drug"])),
                    ("Dosage", f"{course['dose']} · {course['frequency']} · {course['duration_days']} days"),
                    ("Decision", decision)]
            if not accepted:
                reason = course["override_reason"] or "Not recorded"
                if course["override_notes"]:
                    reason += f": {course['override_notes']}"
                rows.append(("Reason for override", reason))
            rows.append(("Prescribed by", f"{course['prescriber']} · "
                                          f"{course['started_on'].strftime('%b %-d, %Y')}"))
            st.html('<div class="fx-verdict fx-verdict-note"><b>A prescription has already been made '
                    'for this patient</b></div>' + "".join(
                        f'<div class="fx-kv"><span>{label}</span><b>{html.escape(value)}</b></div>'
                        for label, value in rows))
            return

        logged = db.assessment_for(clinician["id"], number)
        if logged and logged["outcome"] == "No antibiotic":
            reason = logged["outcome_reason"] + (f": {logged['outcome_notes']}"
                                                 if logged["outcome_notes"] else "")
            st.html('<div class="fx-verdict fx-verdict-note"><b>Decision recorded: no antibiotic</b>'
                    f'<span>{html.escape(reason)} · {html.escape(logged["clinician"])} · '
                    f'{logged["decided_at"].strftime("%b %-d, %Y")}. You can still prescribe below '
                    f'if that changes.</span></div>')

        st.caption(f"Accepting records {policy.drug_label(recommended)}, {regimen_text(recommended)}. "
                   f"It is a usual adult regimen; adjust it for this patient as needed.")
        # Which of the two forms is open for this patient, if either: "override" or "none".
        patient, form = st.session_state.get("deciding", (None, None))
        form = form if patient == number else None
        with st.container(horizontal=True, key="decision_buttons"):
            if st.button("Accept recommendation", key="accept", type="primary", icon=":material/check:"):
                db.save_prescription(clinician["id"], number, recommended, recommended,
                                     policy.USUAL_REGIMENS[recommended])
                st.session_state.pop("deciding", None)
                st.rerun()
            if st.button("Override", key="override", icon=":material/edit:", disabled=form == "override"):
                st.session_state["deciding"] = (number, "override")
                st.rerun()
            if st.button("Do not treat", key="no_treat", icon=":material/block:", disabled=form == "none"):
                st.session_state["deciding"] = (number, "none")
                st.rerun()

        if form == "override":
            others = [drug for drug in drugs if drug != recommended]
            drug = st.selectbox("Prescribe instead", others, format_func=policy.drug_label, index=None,
                                placeholder="Choose an antibiotic", key=f"override_drug_{number}")
            if drug:
                st.caption(f"Usual adult regimen: {regimen_text(drug)}.")
            reason = st.selectbox("Reason for override", db.OVERRIDE_REASONS, index=None,
                                  placeholder="Choose a reason", key=f"override_reason_{number}")
            notes = st.text_area("Notes", placeholder="Add detail (optional)",
                                 key=f"override_notes_{number}")
            ready = drug is not None and reason is not None
        elif form == "none":
            reason = st.selectbox("Reason for not prescribing", db.NO_ANTIBIOTIC_REASONS, index=None,
                                  placeholder="Choose a reason", key=f"none_reason_{number}")
            notes = st.text_area("Notes", placeholder="Add detail (optional)", key=f"none_notes_{number}")
            ready = reason is not None
        else:
            return

        with st.container(horizontal=True, key="override_buttons"):
            if st.button("Save override" if form == "override" else "Save decision", key="override_save",
                         type="primary", icon=":material/check:", disabled=not ready):
                if form == "override":
                    db.save_prescription(clinician["id"], number, recommended, drug,
                                         policy.USUAL_REGIMENS[drug], reason, notes)
                else:
                    db.record_no_antibiotic(clinician["id"], number, reason, notes)
                del st.session_state["deciding"]
                st.rerun()
            if st.button("Cancel", key="override_cancel", type="tertiary"):
                del st.session_state["deciding"]
                st.rerun()


def log_assessment(clinician: dict, record: dict, drugs: list, risks: list, picked: int,
                   target: float):
    """Write this recommendation to the assessment log, once per patient per visit to the page.

    db.log_assessment itself skips patients who already have a prescription and repeats of the
    same recommendation; the check here only saves asking the database on every redraw.
    """
    key = (record["patient_number"], drugs[picked])
    if st.session_state.get("logged") == key:
        return
    checks = {check["name"]: check["level"] for check in prechecks.run(record, drugs[picked])}
    db.log_assessment(clinician["id"], record["patient_number"], drugs[picked], target,
                      {drug: 1 - risk for drug, risk in zip(drugs, risks)},
                      risks[picked] <= 1 - target, checks, data.model_source())
    st.session_state["logged"] = key


def recommendation_panel(model, features, clinician: dict, record: dict):
    predictions = data.predict(model, features, data.patient_features(features, record))
    drugs = list(predictions.index)          # narrowest first
    risks = list(predictions.to_numpy())     # chance of resistance

    with st.container(key="results"):
        target = 90 / 100

        picked = policy.recommend(risks, target)
        log_assessment(clinician, record, drugs, risks, picked, target)
        precheck_strip(record, drugs[picked])
        st.html(hero_html(policy.drug_label(drugs[picked]), 1 - risks[picked],
                          policy.spectrum(drugs[picked]), risks[picked] <= 1 - target, target,
                          policy.explain(risks, drugs, target)))
        decision_card(clinician, record, drugs, drugs[picked])

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
     "The model does not know about allergies, pregnancy, kidney function or symptoms. The "
     "pre-checks above the recommendation flag those from the patient record with a few simple "
     "rules; drug interactions and the site and severity of infection are not checked at all."),
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
        patient_assessment(clinician, record)


def patient_assessment(clinician: dict, record: dict):
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
            target = recommendation_panel(model, features, clinician, record)
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


# ----------------------------------------------------------------------------- frame

st.html(HERE / "styles.css")
try:
    database_ready()
except Exception as error:   # shown to the user: nobody can sign in without the database
    st.error(f"The database ({db.database_kind()}) could not be reached. {error}",
             icon=":material/error:")
    st.stop()

if "clinician" not in st.session_state:
    # Logged out: no sidebar at all (Streamlit otherwise leaves an empty one behind after log out).
    st.html('<style>[data-testid="stSidebar"], [data-testid="stHeader"] { display: none !important; }'
            '</style>')
    st.navigation([st.Page(sign_in_page, title="Log in")], position="hidden").run()
    st.stop()

clinician = st.session_state["clinician"]
assessment = st.Page(assessment_page, title="Clinical Assessment", url_path="assessment",
                     icon=":material/clinical_notes:")
home = st.Page(lambda: dashboard.render(clinician), title="Dashboard", url_path="dashboard",
               icon=":material/dashboard:", default=True)
cultures = st.Page(lambda: culture_results.render(clinician), title="Culture Results",
                   url_path="cultures", icon=":material/biotech:")
follow = st.Page(lambda: follow_up.render(clinician), title="Follow-up", url_path="follow-up",
                 icon=":material/event_repeat:")
stewardship = st.Page(
    lambda: analytics.render(clinician, {"assessment": assessment, "cultures": cultures,
                                         "follow": follow}),
    title="Stewardship Analytics", url_path="analytics", icon=":material/bar_chart:")
current = st.navigation([home, assessment, cultures, follow, stewardship])
if current is not assessment:    # the analytics page sets this just before opening the list
    st.session_state.pop("patient_filter", None)
    st.session_state.pop("deciding", None)
    st.session_state.pop("logged", None)
if current is not assessment:      # coming back to a screen starts at its list
    st.session_state.pop("patient", None)
if current is not cultures:
    st.session_state.pop("culture", None)
if current is not follow:
    st.session_state.pop("follow_up", None)

# The same image for both: Streamlit swaps to icon_image when it thinks the sidebar is collapsed,
# which a browser can still remember from before the sidebar was fixed open.
st.logo(str(LOGO_LIGHT), size="large", icon_image=str(LOGO_LIGHT))
with st.sidebar:
    st.html(f'<div class="fx-sidenote"><b>Signed in</b>{html.escape(clinician["name"])}<br>'
            f'{html.escape(clinician["role"])}</div>')
    if st.button("Log out", key="signout", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.html('<div class="fx-sidenote"><b>Research demo</b>Fabricated patients. '
            'Not for clinical use.</div>')
top_bar(current.title, sample_data=current in (home, assessment, cultures, follow, stewardship))

current.run()

st.html(f"""<div class="fx-footer">
  <img src="{image_data_uri(MARK)}" alt="" />
  <span>Research demo with fabricated patients. <b>Not for clinical use.</b></span>
  <span class="fx-footer-src">Model: {html.escape(data.model_source())} · Database: {db.database_kind()}</span>
</div>""")
