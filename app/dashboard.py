"""The clinician dashboard: the landing page of the platform.

Apart from the signed-in clinician's name, everything shown here is SAMPLE DATA, written out below
(it is not yet read from the database in db.py). There is no patient database yet; when
there is one, replace these constants with functions that read from it and leave `render` alone.
The blank card in the middle column is a placeholder for whatever goes there next.
"""

import datetime
import html

import streamlit as st

SERVICE = "Antimicrobial stewardship service"

# (label, count, colour) for today's worklist
OVERVIEW = [("Awaiting review", 7, "amber"), ("Pending culture results", 12, "teal"),
            ("Follow-up required", 4, "red")]

# Every row leads with its case number; the second line says what is going on.
# (patient, what, level)
ATTENTION = [
    ("PT-00412", "Culture resistant to current therapy · E. coli · on ciprofloxacin", "Critical"),
    ("PT-00389", "De-escalation opportunity · on ceftriaxone · nitrofurantoin susceptible", "High"),
    ("PT-00401", "No option clears the 90% bar · prior ESBL Klebsiella", "High"),
    ("PT-00377", "48-hour reassessment due · empiric nitrofurantoin, day 2", "Medium"),
]

# (patient, what, when)
ACTIVITY = [
    ("PT-00410", "Recommendation accepted · Nitrofurantoin", "2h ago"),
    ("PT-00408", "Recommendation overridden · reason recorded", "4h ago"),
    ("PT-00399", "New assessment · Uncomplicated cystitis", "6h ago"),
    ("PT-00391", "Therapy de-escalated · Ceftriaxone → Cefazolin", "9h ago"),
]

# (culture, specimen and patient, status)
CULTURES = [
    ("CX-2201", "Urine culture · Clean catch · PT-00415", "Preliminary"),
    ("CX-2198", "Urine culture · Catheter · PT-00412", "Pending"),
    ("CX-2195", "Urine culture · Clean catch · PT-00409", "Incubating"),
    ("CX-2190", "Urine culture · Clean catch · PT-00403", "Pending"),
]

# (measure, percent) over the last 7 days
SNAPSHOT = [("Narrow-spectrum first choice", 68), ("Recommendations accepted", 84),
            ("Cultures reviewed within 24h", 91)]

LEVEL_CLASS = {"Critical": "red", "High": "amber", "Medium": "yellow"}
STATUS_CLASS = {"Pending": "navy", "Preliminary": "teal", "Incubating": "violet"}

def _icon(name: str) -> str:
    """A Material Symbols icon by name (the icon font Streamlit already loads for the sidebar)."""
    return f'<span class="fx-msym">{name}</span>'


def _card(icon: str, tone: str, title: str, count, rows: str, fixed: bool = False) -> str:
    badge = f'<span class="fx-count fx-count-{tone}">{count}</span>' if count is not None else ""
    cls = "fx-dcard fx-dcard-fixed" if fixed else "fx-dcard"
    return f"""
    <div class="{cls}">
      <div class="fx-dhead">
        <span class="fx-dicon fx-dicon-{tone}">{_icon(icon)}</span>
        <span class="fx-dtitle">{html.escape(title)}</span>{badge}
      </div>
      <div class="fx-drows">{rows}</div>
    </div>"""


def _row(title: str, sub: str, right: str) -> str:
    return (f'<div class="fx-drow"><div class="fx-drow-text"><b>{html.escape(title)}</b>'
            f'<span>{html.escape(sub)}</span></div>{right}</div>')


def _status(text: str, tone: str) -> str:
    return f'<span class="fx-status fx-status-{tone}">{html.escape(text)}</span>'


def _welcome_html(clinician: dict) -> str:
    today = datetime.date.today().strftime("%A, %B %-d")
    names = clinician["name"].removeprefix("Dr. ").split()
    initials = (names[0][0] + names[-1][0]).upper()
    return f"""
    <div class="fx-dcard fx-welcome">
      <div class="fx-avatar">{html.escape(initials)}</div>
      <div class="fx-welcome-hi">Welcome back,</div>
      <div class="fx-welcome-name">{html.escape(clinician['name'])}</div>
      <span class="fx-badge fx-badge-teal">{html.escape(clinician['role'])}</span>
      <div class="fx-welcome-meta">{today}<br>{SERVICE}</div>
    </div>"""


def _overview_html() -> str:
    tiles = "".join(f'<div class="fx-stat fx-stat-{tone}"><span>{html.escape(label)}</span>'
                    f'<b>{count}</b></div>' for label, count, tone in OVERVIEW)
    return f'<div class="fx-dcard"><div class="fx-dlabel">Today\'s overview</div>{tiles}</div>'


def _snapshot_html() -> str:
    meters = "".join(
        f'<div class="fx-meter"><div class="fx-meter-top"><span>{html.escape(label)}</span>'
        f'<b>{pct}%</b></div><div class="fx-meter-track"><div style="width:{pct}%"></div></div></div>'
        for label, pct in SNAPSHOT)
    return _card("verified_user", "teal", "Stewardship snapshot", None,
                 meters + '<div class="fx-dnote">Last 7 days</div>')


def render(clinician: dict):
    """Draw the dashboard for the signed-in clinician (only the welcome card uses them so far)."""
    with st.container(key="dash"):
        left, right = st.columns([5, 7], gap="small")

    with left:
        st.html(_welcome_html(clinician) + _overview_html())

    with right:
        with st.container(key="dash_cards"):
            attention, cultures = st.columns(2, gap="small")
        with attention:
            st.html(_card("warning", "red", "Requires attention", len(ATTENTION), "".join(
                _row(case, what, _status(level, LEVEL_CLASS[level])) for case, what, level in ATTENTION),
                fixed=True))
        with cultures:
            st.html(_card("science", "violet", "Pending cultures", len(CULTURES), "".join(
                _row(culture, what, _status(status, STATUS_CLASS[status])) for culture, what, status in CULTURES),
                fixed=True))
        st.html(_snapshot_html())
