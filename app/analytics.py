"""Stewardship analytics: totals across the signed-in clinician's patients.

Every figure is counted from rows in the database (prescriptions, cultures, follow-ups) for the
chosen reporting period; `summarise` does the counting and nothing is estimated. Where the
database does not record something a stewardship report would want, the page says so instead of
showing a number (see NOT_LOGGED).

The buttons open the other screens already filtered to the cases behind a figure.
"""

import collections
import datetime

import altair as alt
import pandas as pd
import streamlit as st

import db
import policy
from ui import card

NAVY, TEAL = "#132557", "#009788"
PERIODS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}

# Symptom progression values, best to worst, with the colour each gets in the outcomes chart.
OUTCOME_COLOURS = {"Resolved": "#009788", "Improving": "#7FCFC4", "Unchanged": "#C3CAD8",
                   "Not improving": "#E0A43A", "Worsening": "#C2414B"}
UNRESOLVED = ("Unchanged", "Not improving", "Worsening")

# What a stewardship report would want that the database has no column for yet.
NOT_LOGGED = {
    "decisions": "Not logged yet: cases where no antibiotic cleared the bar, and changes to dose "
                 "or duration alone. A decision is either an acceptance or an override to a "
                 "different antibiotic.",
    "assessments": "Each prescription on record counts as one completed assessment. Opening the "
                   "Clinical Assessment screen is not logged, so assessments that did not end in "
                   "a prescription are not counted.",
}


def percent(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "–"


def summarise(prescriptions: list, cases: list, follow_ups: list, since=None) -> dict:
    """The page's figures, counted from the rows given. `since` (a date) drops anything earlier."""
    if since:
        prescriptions = [row for row in prescriptions if row["started_on"] >= since]
        cases = [case for case in cases if case["collected_on"] >= since]
        follow_ups = [row for row in follow_ups if row["reported_on"] >= since]

    changed = [row for row in prescriptions if row["drug"] != row["recommended_drug"]]
    by_drug = collections.Counter(row["drug"] for row in prescriptions)
    order = list(policy.DRUG_LABELS)        # narrowest first, to break ties in the chart
    resulted = [case for case in cases if case["organism"] is not None]
    mismatches = [case for case in resulted if db.prescription_result(case) == "Resistant"]
    tested, resistant = collections.Counter(), collections.Counter()
    for case in resulted:
        for antibiotic, result in case["susceptibilities"].items():
            tested[antibiotic] += 1
            resistant[antibiotic] += result == "Resistant"
    documented = [row for row in follow_ups if row["symptom_progression"]]
    outcomes = collections.Counter(row["symptom_progression"] for row in documented)

    return {
        "assessments": len(prescriptions),
        "accepted": len(prescriptions) - len(changed),
        "changed": len(changed),
        "changed_patients": [row["patient_number"] for row in changed],
        "changes": collections.Counter((row["recommended_drug"], row["drug"]) for row in changed),
        "reasons": collections.Counter(row.get("override_reason") or "Not recorded"
                                       for row in changed).most_common(),
        "by_drug": sorted(by_drug.items(), key=lambda item: (-item[1], order.index(item[0])
                                                              if item[0] in order else 99)),
        "patients_by_drug": {drug: [row["patient_number"] for row in prescriptions
                                    if row["drug"] == drug] for drug in by_drug},
        "cultures": len(cases),
        "resulted": len(resulted),
        "mismatches": len(mismatches),
        "reviewed": sum(1 for case in resulted if case["review_decision"]),
        "resistance": [(antibiotic, resistant[antibiotic], tested[antibiotic])
                       for antibiotic in sorted(tested, key=lambda a: -resistant[a] / tested[a])],
        "organisms": collections.Counter(case["organism"] for case in resulted).most_common(),
        "follow_ups": len(follow_ups),
        "documented": len(documented),
        "adverse": sum(1 for row in follow_ups if row["adverse_reaction"]),
        "outcomes": [(name, outcomes[name]) for name in OUTCOME_COLOURS if outcomes[name]],
        "unresolved": sum(outcomes[name] for name in UNRESOLVED),
        "needs_reassessment": sum(1 for row in follow_ups if row["status"] == "Needs reassessment"),
    }


# ----------------------------------------------------------------------------- page

def render(clinician: dict, pages: dict):
    """`pages` holds the screens the buttons open: "assessment", "cultures" and "follow"."""
    with st.container(horizontal=True, vertical_alignment="bottom", key="an_head"):
        st.html('<div class="fx-formhead"><div class="fx-title">Stewardship analytics</div>'
                '<div class="fx-sub">Monitor antibiotic utilization, clinical decisions, culture '
                'findings, and treatment follow-up across your patients.</div></div>')
        period = st.selectbox("Reporting period", list(PERIODS), index=1, key="an_period", width=200)
    days = PERIODS[period]
    since = datetime.date.today() - datetime.timedelta(days=days) if days else None
    stats = summarise(db.prescriptions_for(clinician["id"]), db.cultures_for(clinician["id"]),
                      db.follow_ups_for(clinician["id"]), since)

    tiles = [("Total Assessments", "Completed assessments", stats["assessments"], "navy"),
             ("Recommendation Acceptance", "Accepted without modification",
              percent(stats["accepted"], stats["assessments"]), "teal"),
             ("Resistance Mismatch", "Among completed culture results",
              percent(stats["mismatches"], stats["resulted"]), "red"),
             ("Follow-up Completion", "Cases with documented follow-up",
              percent(stats["documented"], stats["follow_ups"]), "amber")]
    st.html('<div class="fx-tiles">' + "".join(
        f'<div class="fx-tile fx-tile-{tone}"><b>{number}</b><span class="fx-tile-name">{label}</span>'
        f'<span>{hint}</span></div>' for label, hint, number, tone in tiles) + '</div>')
    if not stats["assessments"]:
        st.info("Nothing is on record for this reporting period.", icon=":material/info:")
        return

    left, right = st.columns(2, gap="medium")
    with left:
        utilization(stats, pages)
        culture_and_resistance(stats, pages)
    with right:
        recommendation_decisions(stats, pages)
        follow_up_outcomes(stats, pages)


def styled(chart, height: int):
    return (chart.properties(height=height)
            .configure_axis(gridColor="#EDF1F6", domainColor="#D5DCE6", tickColor="#D5DCE6",
                            labelColor="#5B6782", titleColor="#3A4560", titleFontWeight=600,
                            labelFont="Inter", titleFont="Inter", labelLimit=220)
            .configure_view(strokeWidth=0).configure(background="#FFFFFF"))


def open_patients(pages: dict, label: str, numbers: list):
    """Open the Clinical Assessment list showing only these patients."""
    st.session_state["patient_filter"] = {"label": label, "numbers": numbers}
    st.switch_page(pages["assessment"])


def utilization(stats: dict, pages: dict):
    with card("an_rx", "medication", "Antibiotic prescribing distribution",
              "prescriptions by antibiotic"):
        frame = pd.DataFrame([{"Antibiotic": policy.drug_label(drug), "Prescriptions": count}
                              for drug, count in stats["by_drug"]])
        chart = alt.Chart(frame).mark_bar(color=NAVY, size=22).encode(
            y=alt.Y("Antibiotic", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Prescriptions", title="Prescriptions", axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=["Antibiotic", "Prescriptions"])
        st.altair_chart(styled(chart, 46 * len(frame) + 50), width="stretch")

        st.html('<div class="fx-qhead fx-grid-3"><span>Antibiotic</span><span>Prescriptions</span>'
                '<span>Share</span></div>')
        with st.container(key="an_rx_rows"):
            for drug, count in stats["by_drug"]:
                name = policy.drug_label(drug)
                if st.button(f"**{name}** :gray[{count}] :gray[{percent(count, stats['assessments'])}]",
                             key=f"an_rx_{drug}", icon=":material/chevron_right:", width="stretch",
                             help="Open these patients in Clinical Assessment"):
                    open_patients(pages, f"Prescribed {name}", stats["patients_by_drug"][drug])


def recommendation_decisions(stats: dict, pages: dict):
    with card("an_decisions", "rule", "Recommendation decisions",
              "clinician decisions following model recommendations"):
        total = stats["assessments"]
        st.html(
            f'<div class="fx-kv"><span>Accepted without change</span>'
            f'<b>{stats["accepted"]} · {percent(stats["accepted"], total)}</b></div>'
            f'<div class="fx-kv"><span>Different antibiotic prescribed</span>'
            f'<b>{stats["changed"]} · {percent(stats["changed"], total)}</b></div>')
        if stats["changes"]:
            rows = "".join(
                f'<div class="fx-kv"><span>{policy.drug_label(recommended)} → '
                f'{policy.drug_label(prescribed)}</span><b>{count}</b></div>'
                for (recommended, prescribed), count in stats["changes"].most_common())
            st.html('<div class="fx-section">What was changed (recommended → prescribed)</div>' + rows)
            st.html('<div class="fx-section">Reasons for override</div>' + "".join(
                f'<div class="fx-kv"><span>{reason}</span><b>{count}</b></div>'
                for reason, count in stats["reasons"]))
            if st.button("Open changed recommendations in Clinical Assessment", key="an_open_changed",
                         icon=":material/arrow_forward:"):
                open_patients(pages, "Clinician prescribed a different antibiotic than recommended",
                              stats["changed_patients"])
        st.caption(NOT_LOGGED["decisions"])
        st.caption(NOT_LOGGED["assessments"])


def culture_and_resistance(stats: dict, pages: dict):
    with card("an_cultures", "biotech", "Culture & resistance insights"):
        resulted = stats["resulted"]
        st.html(f'<div class="fx-pair"><div><b>{resulted}</b><span>Cultures completed</span></div>'
                f'<div><b>{stats["mismatches"]} · {percent(stats["mismatches"], resulted)}</b>'
                f'<span>Mismatches detected</span></div></div>')
        if not resulted:
            st.caption(f"{stats['cultures']} culture(s) collected; none has a result yet.")
            return
        if stats["mismatches"] and st.button("Open mismatches in Culture Results", key="an_open_mismatch",
                                             icon=":material/arrow_forward:"):
            st.session_state["cx_filter"] = "Resistance mismatch"
            st.switch_page(pages["cultures"])

        frame = pd.DataFrame([{"Antibiotic": policy.drug_label(antibiotic),
                               "Resistant (%)": round(100 * resistant / tested),
                               "Isolates": f"{resistant} of {tested} tested"}
                              for antibiotic, resistant, tested in stats["resistance"]])
        chart = alt.Chart(frame).mark_bar(color=NAVY, size=16).encode(
            y=alt.Y("Antibiotic", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Resistant (%)", scale=alt.Scale(domain=[0, 100]),
                    title="Resistant, among isolates tested for that antibiotic (%)"),
            tooltip=["Antibiotic", "Resistant (%)", "Isolates"])
        st.html('<div class="fx-section">Resistance by antibiotic</div>')
        st.altair_chart(styled(chart, 30 * len(frame) + 50), width="stretch")

        st.html('<div class="fx-section">Most commonly identified organisms</div>' + "".join(
            f'<div class="fx-kv"><span>{organism}</span><b>{count}</b></div>'
            for organism, count in stats["organisms"]))
        st.html('<div class="fx-section">Culture review status</div>'
                f'<div class="fx-kv"><span>Reviewed</span><b>{stats["reviewed"]} · '
                f'{percent(stats["reviewed"], resulted)}</b></div>'
                f'<div class="fx-kv"><span>Awaiting review</span><b>{resulted - stats["reviewed"]} · '
                f'{percent(resulted - stats["reviewed"], resulted)}</b></div>')
        st.caption("Fabricated demo data; resistance rates are not clinical reference values.")


def follow_up_outcomes(stats: dict, pages: dict):
    with card("an_follow", "event_repeat", "Follow-up & treatment monitoring"):
        st.html(f'<div class="fx-pair"><div><b>{percent(stats["documented"], stats["follow_ups"])}</b>'
                f'<span>Follow-up documented ({stats["documented"]} of {stats["follow_ups"]})</span></div>'
                f'<div><b>{stats["adverse"]}</b><span>Adverse reactions</span></div></div>')
        if stats["outcomes"]:
            frame = pd.DataFrame(stats["outcomes"], columns=["Outcome", "Patients"])
            colours = alt.Scale(domain=list(OUTCOME_COLOURS), range=list(OUTCOME_COLOURS.values()))
            chart = alt.Chart(frame).mark_arc(innerRadius=58, outerRadius=100, stroke="#fff",
                                              strokeWidth=2).encode(
                theta=alt.Theta("Patients", stack=True),
                color=alt.Color("Outcome", scale=colours, sort=list(OUTCOME_COLOURS),
                                legend=alt.Legend(orient="right", title=None, labelFontSize=12,
                                                  labelColor="#3A4560", symbolType="square")),
                order=alt.Order("order:Q"), tooltip=["Outcome", "Patients"],
            ).transform_calculate(order=f"indexof({list(OUTCOME_COLOURS)}, datum.Outcome)")
            st.html('<div class="fx-section">Documented symptom progression</div>')
            st.altair_chart(styled(chart, 230), width="stretch")
        st.html(
            f'<div class="fx-kv"><span>Symptoms resolved</span><b>{dict(stats["outcomes"]).get("Resolved", 0)}</b></div>'
            f'<div class="fx-kv"><span>Persistent or worsening symptoms</span><b>{stats["unresolved"]}</b></div>'
            f'<div class="fx-kv"><span>Cases needing reassessment</span><b>{stats["needs_reassessment"]}</b></div>')
        with st.container(horizontal=True, key="an_follow_links"):
            if stats["adverse"] and st.button("Open adverse reactions", key="an_open_adverse",
                                              icon=":material/arrow_forward:"):
                st.session_state["fu_filter"] = "Adverse reaction"
                st.switch_page(pages["follow"])
            if stats["unresolved"] and st.button("Open unresolved symptoms", key="an_open_unresolved",
                                                 icon=":material/arrow_forward:"):
                st.session_state["fu_filter"] = "Unresolved symptoms"
                st.switch_page(pages["follow"])
        st.caption("Outcomes are counted only when documented, not inferred from a course ending.")
