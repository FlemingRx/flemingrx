"""The clinician dashboard: the landing page of the platform.

Everything on it is read from the database for the signed-in clinician's patients, and every
count, row and "See all" opens the screen it came from:

  Today's overview      three counts, each opening the list behind it
  Requires attention    patients whose follow-up needs reassessment  -> Follow-up
  Pending cultures      cultures with no result yet or not yet reviewed -> Culture Results
  Stewardship snapshot  three measures over the last 7 days -> Stewardship Analytics
"""

import datetime
import html

import streamlit as st

import analytics
import db
import policy

SERVICE = "Antimicrobial stewardship service"
ROWS_SHOWN = 4          # each list shows this many; "See all" opens the rest

# How urgent each follow-up reason is on the dashboard, and the colour it is written in.
LEVEL = {"Culture resistance mismatch": "Critical", "Symptoms worsening or not improving": "High",
         "Adverse drug reaction": "High"}
LEVEL_TONE = {"Critical": "red", "High": "orange", "Medium": "gray"}
CULTURE_TONE = {"Pending": "gray", "Ready for review": "orange", "Resistance mismatch": "red"}


def _icon(name: str) -> str:
    """A Material Symbols icon by name (the icon font Streamlit already loads for the sidebar)."""
    return f'<span class="fx-msym">{name}</span>'


def _head(icon: str, tone: str, title: str, count=None) -> str:
    badge = f'<span class="fx-count fx-count-{tone}">{count}</span>' if count is not None else ""
    return (f'<div class="fx-dhead"><span class="fx-dicon fx-dicon-{tone}">{_icon(icon)}</span>'
            f'<span class="fx-dtitle">{html.escape(title)}</span>{badge}</div>')


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


def _open(pages: dict, page: str, **state):
    """Open another screen, first setting what it should show (a filter, or one case)."""
    st.session_state.update(state)
    st.switch_page(pages[page])


def _rows(key: str, rows: list, empty: str):
    """Up to ROWS_SHOWN clickable rows: (id, summary, status, colour, what to do when clicked)."""
    if not rows:
        st.caption(empty)
    with st.container(key=key):
        for number, summary, status, tone, action in rows[:ROWS_SHOWN]:
            # Number, summary and status are separate elements so the row can be laid out as a grid.
            if st.button(f"**{number}** :gray[{summary}] :{tone}[{status}]", key=f"{key}_{number}",
                         width="stretch"):
                action()


def overview(pages: dict, cases: list, follow_ups: list):
    status = [db.culture_status(case) for case in cases]
    tiles = [
        ("Awaiting review", sum(s in ("Ready for review", "Resistance mismatch") for s in status),
         "amber", "cultures", {"cx_filter": "Ready for review"}),
        ("Pending culture results", status.count("Pending"), "teal", "cultures", {"cx_filter": "Pending"}),
        ("Follow-up required", sum(row["status"] == "Needs reassessment" for row in follow_ups),
         "red", "follow", {"fu_filter": "Needs Reassessment"}),
    ]
    with st.container(key="dcard_overview"):
        st.html('<div class="fx-dlabel">Today\'s overview</div>')
        for label, count, tone, page, state in tiles:
            if st.button(f"{label} **{count}**", key=f"dstat_{tone}", width="stretch"):
                _open(pages, page, **state)


def requires_attention(pages: dict, follow_ups: list):
    needing = [row for row in follow_ups if row["status"] == "Needs reassessment"]
    order = list(LEVEL_TONE)
    needing.sort(key=lambda row: order.index(LEVEL.get(row["reason"], "Medium")))
    rows = []
    for row in needing:
        level = LEVEL.get(row["reason"], "Medium")
        number = row["patient_number"]
        summary = f"{row['reason']} · on {policy.drug_label(row['prescription']['drug'])}"
        rows.append((number, summary, level, LEVEL_TONE[level],
                     lambda number=number: _open(pages, "follow", follow_up=number)))
    with st.container(key="dcard_attention"):
        st.html(_head("warning", "red", "Requires attention", len(rows)))
        _rows("drows_attention", rows, "No patients need reassessment.")
        if st.button("See all", key="dsee_attention", icon=":material/arrow_forward:", type="tertiary"):
            _open(pages, "follow", fu_filter="Needs Reassessment")


def pending_cultures(pages: dict, cases: list):
    order = list(CULTURE_TONE)
    waiting = [(case, db.culture_status(case)) for case in cases]
    waiting = sorted((pair for pair in waiting if pair[1] != "Reviewed"),
                     key=lambda pair: order.index(pair[1]))
    rows = []
    for case, status in waiting:
        number = case["culture_number"]
        summary = f"{case['patient_number']} · {case['organism'] or 'Result awaited'}"
        rows.append((number, summary, status, CULTURE_TONE[status],
                     lambda number=number: _open(pages, "cultures", culture=number)))
    with st.container(key="dcard_cultures"):
        st.html(_head("science", "violet", "Pending cultures", len(rows)))
        _rows("drows_cultures", rows, "No cultures are waiting for a result or a review.")
        if st.button("See all", key="dsee_cultures", icon=":material/arrow_forward:", type="tertiary"):
            _open(pages, "cultures", cx_filter="All")


def snapshot(pages: dict, stats: dict):
    narrow = sum(count for drug, count in stats["by_drug"] if policy.spectrum(drug) == "Narrow")
    measures = [("Narrow-spectrum antibiotic prescribed", narrow, stats["prescriptions"]),
                # both halves from the assessment log, so they are always counting the same cases
                ("Recommendations accepted", dict(stats["decisions"])["Accepted"], stats["decided"]),
                ("Culture results reviewed", stats["reviewed"], stats["resulted"])]
    with st.container(key="dcard_snapshot"):
        with st.container(horizontal=True, vertical_alignment="center", key="dsnap_head"):
            st.html(_head("verified_user", "teal", "Stewardship snapshot"))
            if st.button("See all", key="dsee_snapshot", icon=":material/arrow_forward:",
                         type="tertiary"):
                _open(pages, "analytics")
        st.html("".join(
            f'<div class="fx-meter"><div class="fx-meter-top"><span>{label}</span>'
            f'<b>{analytics.percent(part, whole)} <i>{part} of {whole}</i></b></div>'
            f'<div class="fx-meter-track"><div style="width:{100 * part / whole if whole else 0:.0f}%">'
            f'</div></div></div>' for label, part, whole in measures)
            + '<div class="fx-dnote">Last 7 days · your patients</div>')


def render(clinician: dict, pages: dict):
    """Draw the dashboard. `pages` holds the screens it links to: "cultures", "follow", "analytics"."""
    cases = db.cultures_for(clinician["id"])
    follow_ups = db.follow_ups_for(clinician["id"])
    week_ago = datetime.date.today() - datetime.timedelta(days=7)
    stats = analytics.summarise(db.prescriptions_for(clinician["id"]), cases, follow_ups,
                                db.assessments_for(clinician["id"]), week_ago)

    with st.container(key="dash"):
        left, right = st.columns([5, 7], gap="small")
    with left:
        st.html(_welcome_html(clinician))
        overview(pages, cases, follow_ups)
    with right:
        with st.container(key="dash_cards"):
            attention, cultures = st.columns(2, gap="small")
        with attention:
            requires_attention(pages, follow_ups)
        with cultures:
            pending_cultures(pages, cases)
        snapshot(pages, stats)
