"""Building blocks shared by the screens: cards, and the read-only view of a patient's record."""

import html

import streamlit as st

import data
import policy


def card(key: str, icon: str, title: str, hint: str | None = None):
    """A white card with a header row, like the sections of a clinical form. Returns its body."""
    box = st.container(border=True, key=f"card_{key}")
    with box.container(key=f"cardhead_{key}", horizontal=True, vertical_alignment="center"):
        st.markdown(f":material/{icon}: **{title}**")
        if hint:
            st.caption(hint)
    return box


def last_antibiotic_text(days: int) -> str:
    if days >= data.NO_RECENT_ANTIBIOTIC:
        return "None in the past year"
    return f"About {days} days ago"


def names(slugs, label=data.pretty) -> str:
    return ", ".join(label(slug) for slug in slugs) or "None"


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
