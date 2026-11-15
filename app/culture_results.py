"""Culture results: the clinician's cultures as a worklist, and one page per culture.

The list shows four counts and every culture for the clinician's patients. Opening one shows the
patient's details, what the lab found, how the current prescription compares with it, and a place
to record the clinician's review. Nothing here changes a prescription: a mismatch is a prompt for
the clinician to reassess, and the review only records what they decided.
"""

import html

import streamlit as st

import db
import policy
from ui import card, patient_record, record_card

# Worklist order, and the colour each status is written in (list rows use Streamlit's colour names).
STATUSES = {"Resistance mismatch": "red", "Ready for review": "orange", "Pending": "gray",
            "Reviewed": "green"}
RESULT_CLASS = {"Susceptible": "ok", "Intermediate": "warn", "Resistant": "bad"}
DECISION_HELP = {
    "Compatible": "Current prescription appears appropriate based on susceptibility.",
    "Reassessment needed": "Further clinical evaluation is warranted.",
    "Other": "Document another interpretation or action.",
}


def day(date) -> str:
    return date.strftime("%b %-d, %Y")


def render(clinician: dict):
    number = st.session_state.get("culture")
    # culture_for only returns cultures of this clinician's patients.
    case = db.culture_for(clinician["id"], number) if number else None
    if case is None:
        case_list(clinician)
    else:
        case_page(clinician, case)


# ----------------------------------------------------------------------------- the list

def case_list(clinician: dict):
    cases = db.cultures_for(clinician["id"])
    status = {case["culture_number"]: db.culture_status(case) for case in cases}
    count = lambda *names: sum(1 for value in status.values() if value in names)   # noqa: E731

    st.html('<div class="fx-formhead"><div class="fx-title">Culture results</div>'
            '<div class="fx-sub">Urine cultures for your patients. Select one to see the result '
            'and record your review.</div></div>')
    # A mismatch is also waiting for review, so it is counted in both of the middle two.
    tiles = [("Pending", count("Pending"), "navy"),
             ("Ready for review", count("Ready for review", "Resistance mismatch"), "amber"),
             ("Resistance mismatches", count("Resistance mismatch"), "red"),
             ("Reviewed", count("Reviewed"), "teal")]
    st.html('<div class="fx-tiles">' + "".join(
        f'<div class="fx-tile fx-tile-{tone}"><b>{number}</b><span>{label}</span></div>'
        for label, number, tone in tiles) + '</div>')

    if not cases:
        st.info("There are no cultures for your patients yet.", icon=":material/info:")
        return
    search = st.text_input("Search culture or patient ID", icon=":material/search:",
                           placeholder="Search culture or patient ID...",
                           label_visibility="collapsed", key="culture_search", width=280)
    wanted = search.strip().lower()
    shown = [case for case in cases if wanted in case["culture_number"].lower()
             or wanted in case["patient_number"].lower()]
    if not shown:
        st.caption("No cultures match.")
    order = list(STATUSES)
    with st.container(key="caselist"):
        for case in sorted(shown, key=lambda case: order.index(status[case["culture_number"]])):
            number, state = case["culture_number"], status[case["culture_number"]]
            about = " · ".join([case["patient_number"], case["organism"] or "Result awaited",
                                f"collected {day(case['collected_on'])}"])
            if st.button(f"**{number}** :gray[{about}] :{STATUSES[state]}[{state}]",
                         key=f"open_{number}", icon=":material/chevron_right:", width="stretch"):
                st.session_state["culture"] = number
                st.rerun()


# ----------------------------------------------------------------------------- one culture

def case_page(clinician: dict, case: dict):
    if st.button("All cultures", key="back", icon=":material/arrow_back:", type="tertiary"):
        del st.session_state["culture"]
        st.rerun()
    state = db.culture_status(case)
    st.html(f'<div class="fx-formhead"><div class="fx-title">{html.escape(case["culture_number"])}'
            f'</div><div class="fx-sub">Urine culture for {html.escape(case["patient_number"])} · '
            f'<b class="fx-state-{STATUSES[state]}">{state}</b></div></div>')

    patient_column, culture_column = st.columns([5, 7], gap="large")
    with patient_column:
        st.html('<div class="fx-section">Patient details</div>')
        patient_record(case["patient"])
    with culture_column:
        st.html('<div class="fx-section">Culture</div>')
        culture_information(case)
        susceptibility_results(case)
        prescription_comparison(case)
        clinician_review(clinician, case)


def culture_information(case: dict):
    resulted = case["organism"] is not None
    record_card("cx_info", "biotech", "Culture information", [
        ("Patient ID", case["patient_number"]),
        ("Specimen", case["specimen"]),
        ("Collection date", day(case["collected_on"])),
        ("Organism", case["organism"] if resulted else "Result awaited"),
        ("Colony count", f"{case['colony_count']:,} CFU/mL" if resulted else "Result awaited"),
    ])


def panel_order(antibiotics) -> list:
    """Antibiotics the model doesn't cover first, then the model's, narrowest to broadest."""
    known = list(policy.DRUG_LABELS)
    return sorted(antibiotics, key=lambda name: (name in known, known.index(name) if name in known
                                                 else 0, name))


def susceptibility_results(case: dict):
    with card("cx_panel", "science", "Antibiotic susceptibility results"):
        results = case["susceptibilities"]
        if not results:
            st.caption("The lab has not reported yet.")
            return
        prescribed = case["prescription"]["drug"] if case["prescription"] else None
        rows = "".join(
            f'<div class="fx-kv"><span>{html.escape(policy.drug_label(name))}'
            f'{"<i>current prescription</i>" if name == prescribed else ""}</span>'
            f'<b class="fx-res-{RESULT_CLASS[results[name]]}">{results[name]}</b></div>'
            for name in panel_order(results))
        st.html(rows)


def prescription_comparison(case: dict):
    with card("cx_compare", "compare_arrows", "Current prescription comparison"):
        course = case["prescription"]
        if course is None:
            st.caption("No prescription is recorded for this patient.")
            return
        followed = course["drug"] == course["recommended_drug"]
        st.html(f"""
        <div class="fx-compare">
          <div><span class="fx-eyebrow-dark">Initial model recommendation</span>
            <b>{html.escape(policy.drug_label(course["recommended_drug"]))}</b>
            <small>Narrowest option that cleared a 90% chance of working</small></div>
          <div><span class="fx-eyebrow-dark">Current prescription</span>
            <b>{html.escape(policy.drug_label(course["drug"]))}</b>
            <small>{html.escape(course["dose"])} · {html.escape(course["frequency"])} ·
              {course["duration_days"]} days · started {day(course["started_on"])}<br>
              {"Clinician accepted the recommendation" if followed
               else "Clinician chose a different antibiotic"}</small></div>
        </div>""")

        result = db.prescription_result(case)
        if case["organism"] is None:
            verdict = ("note", "Awaiting culture result",
                       "The prescription can be compared once the lab reports.")
        elif result == "Susceptible":
            verdict = ("ok", "Susceptible according to culture",
                       "The identified organism is reported susceptible to the prescribed antibiotic.")
        elif result == "Resistant":
            verdict = ("bad", "Potential resistance mismatch",
                       "The identified organism is resistant to the patient's currently prescribed "
                       "antibiotic. Clinician reassessment recommended. This is a decision-support "
                       "alert, not an automatic prescription change.")
        elif result == "Intermediate":
            verdict = ("warn", "Intermediate susceptibility",
                       "The identified organism is reported intermediate to the prescribed "
                       "antibiotic. Clinician reassessment recommended.")
        else:
            verdict = ("warn", "Prescribed antibiotic not tested",
                       "The lab's panel does not include the prescribed antibiotic, so the culture "
                       "cannot confirm it.")
        tone, title, body = verdict
        st.html(f'<div class="fx-verdict fx-verdict-{tone}"><b>{title}</b><span>{body}</span></div>')


def clinician_review(clinician: dict, case: dict):
    with card("cx_review", "rate_review", "Clinician review"):
        if case["organism"] is None:
            st.caption("A review can be recorded once the lab reports.")
            return
        if case["review_decision"]:
            st.html(
                f'<div class="fx-kv"><span>Review decision</span><b>{case["review_decision"]}</b></div>'
                f'<div class="fx-kv"><span>Clinical notes</span>'
                f'<b>{html.escape(case["review_notes"] or "None recorded")}</b></div>'
                f'<div class="fx-kv"><span>Reviewed by</span><b>{html.escape(case.get("reviewer", ""))}'
                f' · {case["reviewed_at"].strftime("%b %-d, %Y %H:%M")}</b></div>')
            return

        st.caption("Record your interpretation and next steps.")
        decision = st.radio("Review decision", db.REVIEW_DECISIONS, index=None, key="review_decision",
                            captions=[DECISION_HELP[name] for name in db.REVIEW_DECISIONS])
        notes = st.text_area("Clinical notes", placeholder="Enter review notes...", key="review_notes")
        if st.button("Mark as reviewed", key="review_save", type="primary", icon=":material/check:",
                     disabled=decision is None, width="stretch"):
            db.review_culture(clinician["id"], case["culture_number"], decision, notes)
            st.rerun()
