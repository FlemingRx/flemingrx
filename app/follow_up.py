"""Follow-up: which patients need another look after a prescription, and one page per patient.

The queue shows four counts and one row per patient. Opening a row shows the patient's details and
five parts: the original treatment, why they are in the queue, what has been documented since, a
summary of all of it, and a form for the clinician's reassessment.

The summary and its "suggested next step" are fixed wording chosen by simple rules from what is on
record (see reassessment_summary). No model writes them, and nothing here changes a prescription.
"""

import html

import streamlit as st

import db
import policy
from ui import card, patient_record, record_card

# Queue order, and the colour each status is written in.
STATUS_TONE = {"Needs reassessment": "red", "Awaiting follow-up": "blue", "Resolved": "green"}
FILTERS = {"All": None, "Needs Reassessment": "Needs reassessment", "Awaiting": "Awaiting follow-up",
           "Resolved": "Resolved"}
NOT_YET = "Not yet documented"

NEXT_STEP = {
    "Culture resistance mismatch":
        "Reassess the patient's clinical status and review the culture susceptibility findings "
        "before determining whether treatment should continue or change.",
    "Symptoms worsening or not improving":
        "Reassess the patient's clinical status and review the culture susceptibility findings "
        "before determining whether treatment should continue or change.",
    "Adverse drug reaction":
        "Review the documented reaction and assess whether the current antibiotic should continue.",
    "Treatment completion":
        "Confirm the outcome of treatment and record whether symptoms have resolved.",
    "Other clinical concern":
        "Review the documented concern and decide whether further evaluation is needed.",
    "Follow-up pending":
        "No follow-up information is available yet. Review once it has been documented.",
}


def day(date) -> str:
    return date.strftime("%b %-d, %Y")


def prescription_text(course: dict) -> str:
    return f"{policy.drug_label(course['drug'])} {course['dose']}"


def render(clinician: dict):
    number = st.session_state.get("follow_up")
    # follow_up_for only returns patients of this clinician.
    row = db.follow_up_for(clinician["id"], number) if number else None
    if row is None:
        queue(clinician)
    else:
        patient_page(clinician, row)


# ----------------------------------------------------------------------------- the queue

def queue(clinician: dict):
    rows = db.follow_ups_for(clinician["id"])
    count = lambda status: sum(1 for row in rows if row["status"] == status)   # noqa: E731

    st.html('<div class="fx-formhead"><div class="fx-title">Follow-up</div>'
            '<div class="fx-sub">Monitor treatment progress and identify patients requiring '
            'clinical reassessment.</div></div>')
    tiles = [("Needs Reassessment", "Requires clinician review", count("Needs reassessment"), "red"),
             ("Awaiting Follow-up", "Information not yet available", count("Awaiting follow-up"), "navy"),
             ("Adverse Reactions", "Documented medication concerns",
              sum(1 for row in rows if row["adverse_reaction"]), "amber"),
             ("Resolved", "Follow-up completed", count("Resolved"), "teal")]
    st.html('<div class="fx-tiles">' + "".join(
        f'<div class="fx-tile fx-tile-{tone}"><b>{number}</b><span class="fx-tile-name">{label}</span>'
        f'<span>{hint}</span></div>' for label, hint, number, tone in tiles) + '</div>')

    # with card("fu_queue", "checklist", "Follow-up queue"):
    with st.container(horizontal=True, vertical_alignment="bottom", key="fu_controls"):
        search = st.text_input("Search patient ID", placeholder="Search patient ID...",
                                icon=":material/search:", label_visibility="collapsed",
                                key="fu_search", width=280)
        chosen = st.segmented_control("Status", list(FILTERS), default="All", required=True,
                                        label_visibility="collapsed", key="fu_filter")
    wanted = FILTERS[chosen]
    shown = [row for row in rows
                if (wanted is None or row["status"] == wanted)
                and search.strip().lower() in row["patient_number"].lower()]
    order = list(STATUS_TONE)
    shown.sort(key=lambda row: order.index(row["status"]))      # stable: newest first within each

    st.html('<div class="fx-qhead"><span>Patient ID</span><span>Original prescription</span>'
            '<span>Follow-up reason</span><span>Last updated</span><span>Follow-up status</span></div>')
    if not shown:
        st.caption("No patients match.")
    with st.container(key="fu_rows"):
        for row in shown:
            number = row["patient_number"]
            # Each cell is its own element so the row can be laid out as a grid (styles.css).
            label = (f"**{number}** :gray[{prescription_text(row['prescription'])}] "
                        f":gray[{row['reason']}] :gray[{day(row['updated_at'])}] "
                        f":{STATUS_TONE[row['status']]}[{row['status']}]")
            if st.button(label, key=f"fu_open_{number}", icon=":material/chevron_right:",
                            width="stretch"):
                st.session_state["follow_up"] = number
                st.rerun()


# ----------------------------------------------------------------------------- one patient

def patient_page(clinician: dict, row: dict):
    if st.button("Follow-up queue", key="back", icon=":material/arrow_back:", type="tertiary"):
        del st.session_state["follow_up"]
        st.rerun()
    st.html(f'<div class="fx-formhead"><div class="fx-title">{html.escape(row["patient_number"])}'
            f'</div><div class="fx-sub">Follow-up · '
            f'<b class="fx-state-{STATUS_TONE[row["status"]]}">{row["status"]}</b></div></div>')

    patient_column, follow_column = st.columns([5, 7], gap="large")
    with patient_column:
        st.html('<div class="fx-section">Patient details</div>')
        patient_record(row["patient"])
    with follow_column:
        st.html('<div class="fx-section">Follow-up</div>')
        original_treatment(row)
        reason_for_follow_up(row)
        follow_up_information(row)
        reassessment_card(row)
        clinician_decision(clinician, row)


def original_treatment(row: dict):
    course = row["prescription"]
    record_card("fu_original", "medication", "Original treatment", [
        ("Assessment date", day(course["started_on"])),
        ("Diagnosis", "UTI"),
        ("Prescribed antibiotic", policy.drug_label(course["drug"])),
        ("Dosage", f"{course['dose']} {course['frequency'].lower()}"),
        ("Duration", f"{course['duration_days']} days"),
    ])


def reason_for_follow_up(row: dict):
    title = "Reason for reassessment" if row["status"] == "Needs reassessment" else "Reason for follow-up"
    with card("fu_reason", "warning", title):
        st.html(f'<div class="fx-reason"><b class="fx-state-{STATUS_TONE[row["status"]]}">'
                f'{html.escape(row["reason"])}</b><p>{html.escape(row["reason_detail"])}</p>'
                f'<small>{html.escape(row["source"])} · {day(row["reported_on"])}</small></div>')


def culture_text(row: dict) -> str:
    culture = row["culture"]
    if culture is None:
        return "None on record"
    if culture["organism"] is None:
        return f"Pending ({culture['culture_number']})"
    return f"Available ({culture['culture_number']}, {culture['organism']})"


def follow_up_information(row: dict):
    record_card("fu_info", "clinical_notes", "Follow-up information", [
        ("Symptom progression", row["symptom_progression"] or NOT_YET),
        ("Treatment adherence", row["treatment_adherence"] or NOT_YET),
        ("Adverse reactions", row["adverse_reaction"] or
         ("None reported" if row["symptom_progression"] else NOT_YET)),
        ("Culture results", culture_text(row)),
        ("Additional notes", row["additional_notes"] or "None"),
    ])


def reassessment_summary(row: dict) -> tuple[list[str], str]:
    """What is on record, as short sentences, and the suggested next step for the reason given."""
    course, culture = row["prescription"], row["culture"]
    drug = policy.drug_label(course["drug"])
    points = [f"Original prescription: {drug} {course['dose']} {course['frequency'].lower()} "
              f"for {course['duration_days']} days."]
    if culture is None or culture["organism"] is None:
        points.append("Culture results are not yet available.")
    else:
        result = db.prescription_result({**culture, "prescription": course})
        finding = f"reported {result.lower()} to {drug}" if result else f"not tested against {drug}"
        points.append(f"Culture {culture['culture_number']} grew {culture['organism']}, {finding}.")
    points.append(f"Symptom progression documented as: {row['symptom_progression'].lower()}."
                  if row["symptom_progression"] else "Symptom progression is not yet documented.")
    if row["treatment_adherence"]:
        points.append(f"Treatment adherence: {row['treatment_adherence'].lower()}.")
    points.append(f"Adverse reaction documented: {row['adverse_reaction']}."
                  if row["adverse_reaction"] else "No adverse reactions documented.")
    patient = row["patient"]
    if patient["resistant_to"]:
        earlier = ", ".join(policy.drug_label(name) for name in patient["resistant_to"])
        points.append(f"Earlier cultures in the past 12 months were resistant to: {earlier}.")

    if row["status"] == "Resolved":
        return points, "Follow-up is recorded as resolved. No further action is suggested."
    return points, NEXT_STEP[row["reason"]]


def reassessment_card(row: dict):
    tag = {"Needs reassessment": "Clinician review recommended",
           "Awaiting follow-up": "Awaiting follow-up information",
           "Resolved": "Follow-up completed"}[row["status"]]
    points, next_step = reassessment_summary(row)
    with card("fu_summary", "summarize", "Reassessment summary"):
        st.html(f'<div class="fx-summary"><b class="fx-state-{STATUS_TONE[row["status"]]}">{tag}</b>'
                f'<ul>{"".join(f"<li>{html.escape(point)}</li>" for point in points)}</ul>'
                f'<div class="fx-next"><span class="fx-eyebrow-dark">Suggested next step</span>'
                f'<p>{html.escape(next_step)}</p></div>'
                f'<small>Decision-support summary; final clinical decisions remain with the '
                f'clinician.</small></div>')


def clinician_decision(clinician: dict, row: dict):
    with card("fu_decision", "rate_review", "Clinician decision"):
        if row["decision"]:
            st.html(
                f'<div class="fx-kv"><span>Last clinical decision</span><b>{row["decision"]}</b></div>'
                f'<div class="fx-kv"><span>Clinical notes</span>'
                f'<b>{html.escape(row["decision_notes"] or "None recorded")}</b></div>'
                f'<div class="fx-kv"><span>Recorded by</span><b>{html.escape(row.get("decider", ""))}'
                f' · {row["decided_at"].strftime("%b %-d, %Y %H:%M")}</b></div>')
        if row["status"] == "Resolved":
            return

        st.caption("Document the outcome of your reassessment.")
        version = row["decided_at"]      # in the keys, so the form comes back empty after a save
        decision = st.radio("Clinical decision", db.FOLLOW_UP_DECISIONS, index=None,
                            key=f"fu_decision_{version}")
        notes = st.text_area("Clinical notes", placeholder="Enter reassessment notes...",
                             key=f"fu_notes_{version}")
        outcome = st.radio("Follow-up status", list(db.FOLLOW_UP_OUTCOMES), index=None,
                           key=f"fu_outcome_{version}")
        if st.button("Save reassessment", key="fu_save", type="primary", icon=":material/check:",
                     disabled=decision is None or outcome is None, width="stretch"):
            db.save_reassessment(clinician["id"], row["patient_number"], decision, notes, outcome)
            st.rerun()
