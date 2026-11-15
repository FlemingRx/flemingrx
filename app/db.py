"""The database: clinicians, patients, and which clinician looks after which patient.

The tables:
  clinicians          one row per login (the password is stored as a salted hash, never as text)
  patients            one row per patient, holding the details the model needs
  clinician_patients  one row per (clinician, patient) pair, so a patient can have several clinicians
  prescriptions       what the model recommended for a patient and what the clinician prescribed
  cultures            one row per urine culture: what grew, and the clinician's review of the result
  susceptibilities    one row per culture and antibiotic tested: susceptible, intermediate or resistant
  follow_ups          one row per patient being followed after a prescription: why, what has been
                      documented since, and the clinician's reassessment

Where the database lives:
  Databricks Lakebase (Postgres), by default, both deployed and on a laptop. There is no stored
      password: each new connection signs in with a short-lived token from Databricks. The compute
      to use is ENDPOINT_NAME (default below). A deployed Databricks App with a database resource
      is given PGHOST and PGUSER; on a laptop they are looked up with your Databricks CLI sign-in.
  DATABASE_URL, if set, is used instead: postgresql://user:password@host:5432/dbname for another
      Postgres, or sqlite:///flemingrx.db for a file on this machine with no Databricks needed.
The same code runs on all of them. The first time it finds an empty database it creates the tables
and fills them with the fabricated demo data in seed.py.
"""

import datetime
import functools
import hashlib
import hmac
import os
import secrets

import sqlalchemy as sa

metadata = sa.MetaData()

clinicians = sa.Table(
    "clinicians", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("username", sa.String(64), nullable=False, unique=True),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("role", sa.String(120), nullable=False),
    sa.Column("password_hash", sa.String(255), nullable=False),
)

# The patient columns are the details the assessment used to ask for in a form. The four list
# columns hold short names that match the model's features, e.g. "ciprofloxacin", "klebsiella".
patients = sa.Table(
    "patients", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("patient_number", sa.String(32), nullable=False, unique=True),
    sa.Column("age_band", sa.String(16), nullable=False),            # a key of data.AGE_BANDS
    sa.Column("sex", sa.String(16), nullable=False),                 # a key of data.SEX_CODES
    sa.Column("care_setting", sa.String(64), nullable=False),        # a key of data.SETTINGS
    sa.Column("nursing_home", sa.Boolean, nullable=False, default=False),
    sa.Column("resistant_to", sa.JSON, nullable=False, default=list),          # past 12 months
    sa.Column("other_resistant_count", sa.Integer, nullable=False, default=0),
    sa.Column("recent_antibiotic_classes", sa.JSON, nullable=False, default=list),   # last 90 days
    sa.Column("days_since_last_antibiotic", sa.Integer, nullable=False, default=366),
    sa.Column("prior_organisms", sa.JSON, nullable=False, default=list),       # past year
    sa.Column("recent_procedures", sa.JSON, nullable=False, default=list),     # last 90 days
)

clinician_patients = sa.Table(
    "clinician_patients", metadata,
    sa.Column("clinician_id", sa.ForeignKey("clinicians.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("patient_id", sa.ForeignKey("patients.id", ondelete="CASCADE"), primary_key=True),
)

# One row per treatment course. `recommended_drug` is what the model suggested at the time and
# `drug` is what the clinician prescribed; they differ when the clinician overrode the model.
# Drug names are the model's short names, e.g. "nitrofurantoin" (see policy.DRUG_LABELS).
prescriptions = sa.Table(
    "prescriptions", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("patient_id", sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
    sa.Column("prescribed_by", sa.ForeignKey("clinicians.id"), nullable=False),
    sa.Column("recommended_drug", sa.String(64), nullable=False),
    sa.Column("drug", sa.String(64), nullable=False),
    sa.Column("dose", sa.String(64), nullable=False),              # e.g. "100 mg"
    sa.Column("frequency", sa.String(64), nullable=False),         # e.g. "Twice daily"
    sa.Column("duration_days", sa.Integer, nullable=False),
    sa.Column("started_on", sa.Date, nullable=False),
)

# A culture is "pending" until the lab reports (resulted_on, organism and colony count are empty
# until then). The four review columns stay empty until a clinician marks the result as reviewed.
cultures = sa.Table(
    "cultures", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("culture_number", sa.String(32), nullable=False, unique=True),
    sa.Column("patient_id", sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
    sa.Column("specimen", sa.String(64), nullable=False),
    sa.Column("collected_on", sa.Date, nullable=False),
    sa.Column("resulted_on", sa.Date),
    sa.Column("organism", sa.String(120)),
    sa.Column("colony_count", sa.Integer),                         # CFU/mL
    sa.Column("review_decision", sa.String(32)),                   # one of REVIEW_DECISIONS
    sa.Column("review_notes", sa.Text),
    sa.Column("reviewed_by", sa.ForeignKey("clinicians.id")),
    sa.Column("reviewed_at", sa.DateTime),
)

susceptibilities = sa.Table(
    "susceptibilities", metadata,
    sa.Column("culture_id", sa.ForeignKey("cultures.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("antibiotic", sa.String(64), primary_key=True),
    sa.Column("result", sa.String(16), nullable=False),            # one of RESULTS
)

# One row per patient in the follow-up queue. The first block says why they are in it and where
# that came from (a culture or a documented encounter). The second holds what was documented after
# the prescription; in a hospital it would be imported from the record. The third is the
# clinician's reassessment, empty until they save one. `status` is where the row sits in the queue.
follow_ups = sa.Table(
    "follow_ups", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("patient_id", sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False,
              unique=True),
    sa.Column("prescription_id", sa.ForeignKey("prescriptions.id", ondelete="CASCADE"),
              nullable=False),
    sa.Column("reason", sa.String(64), nullable=False),            # one of FOLLOW_UP_REASONS
    sa.Column("reason_detail", sa.Text, nullable=False),
    sa.Column("source", sa.String(120), nullable=False),           # the culture or encounter
    sa.Column("reported_on", sa.Date, nullable=False),
    sa.Column("status", sa.String(32), nullable=False),            # one of FOLLOW_UP_STATUSES
    sa.Column("updated_at", sa.DateTime, nullable=False),

    sa.Column("symptom_progression", sa.String(64)),
    sa.Column("treatment_adherence", sa.String(64)),
    sa.Column("adverse_reaction", sa.String(255)),                 # empty = none reported
    sa.Column("additional_notes", sa.Text),

    sa.Column("decision", sa.String(64)),                          # one of FOLLOW_UP_DECISIONS
    sa.Column("decision_notes", sa.Text),
    sa.Column("decided_by", sa.ForeignKey("clinicians.id")),
    sa.Column("decided_at", sa.DateTime),
)

RESULTS = ("Susceptible", "Intermediate", "Resistant")
FOLLOW_UP_REASONS = ("Culture resistance mismatch", "Symptoms worsening or not improving",
                     "Adverse drug reaction", "Treatment completion", "Other clinical concern",
                     "Follow-up pending")
FOLLOW_UP_STATUSES = ("Needs reassessment", "Awaiting follow-up", "Resolved")
FOLLOW_UP_DECISIONS = ("Continue current treatment", "Modify treatment",
                       "Escalate for further evaluation", "Other / additional monitoring")
# What the clinician picks when saving a reassessment, and the queue status it leads to.
FOLLOW_UP_OUTCOMES = {"Resolved": "Resolved", "Further monitoring required": "Awaiting follow-up",
                      "Reassessment pending": "Needs reassessment"}
REVIEW_DECISIONS = ("Compatible", "Reassessment needed", "Other")


# ----------------------------------------------------------------------------- connection

# On Lakebase the tables go in a schema of their own. The app's login is allowed to create
# schemas in the database, but not necessarily tables in the shared "public" one.
LAKEBASE_SCHEMA = os.getenv("FLEMINGRX_DB_SCHEMA", "flemingrx")


# The Lakebase compute this project uses (Lakebase app -> branch -> Computes -> Get ID).
DEFAULT_ENDPOINT = "projects/flemingrx/branches/production/endpoints/primary"


def endpoint_name() -> str:
    return os.getenv("ENDPOINT_NAME") or DEFAULT_ENDPOINT


def on_lakebase() -> bool:
    return not os.getenv("DATABASE_URL")


def _workspace():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _lakebase_host_and_user() -> tuple[str, str]:
    """Where to connect and who as: given to a deployed app, looked up from Databricks on a laptop."""
    host, user = os.getenv("PGHOST"), os.getenv("PGUSER")
    if not host:
        host = _workspace().postgres.get_endpoint(name=endpoint_name()).status.hosts.host
    if not user:
        user = _workspace().current_user.me().user_name
    return host, user


def database_url() -> sa.URL | str:
    url = os.getenv("DATABASE_URL")
    if url:
        # Hosting services hand out postgres:// or postgresql:// addresses; name the driver we install.
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url
    host, user = _lakebase_host_and_user()
    return sa.URL.create(
        "postgresql+psycopg", username=user, host=host, port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "databricks_postgres"),
        query={"sslmode": os.getenv("PGSSLMODE", "require"),
               "options": f"-csearch_path={LAKEBASE_SCHEMA}"})


def database_kind() -> str:
    """Which database is in use, for showing in the app."""
    if on_lakebase():
        return "Databricks Lakebase (Postgres)"
    return "SQLite (local file)" if os.environ["DATABASE_URL"].startswith("sqlite") else "Postgres"


def _lakebase_token() -> str:
    """A fresh sign-in token for Lakebase. They last an hour, so each new connection asks for one."""
    return _workspace().postgres.generate_database_credential(endpoint=endpoint_name()).token


@functools.lru_cache(maxsize=None)
def engine(url: str | None = None) -> sa.Engine:
    if url or not on_lakebase():
        return sa.create_engine(url or database_url(), pool_pre_ping=True)

    # Drop pooled connections before their token's hour is up, and sign each new one in afresh.
    db = sa.create_engine(database_url(), pool_pre_ping=True, pool_recycle=45 * 60)

    @sa.event.listens_for(db, "do_connect")
    def add_token(dialect, connection_record, connection_args, connection_kwargs):
        connection_kwargs["password"] = _lakebase_token()

    return db


def init(url: str | None = None) -> None:
    """Create the tables if they are missing, and add the demo data if there are no clinicians."""
    import seed

    db = engine(url)
    if not url and on_lakebase():
        with db.begin() as connection:
            connection.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{LAKEBASE_SCHEMA}"'))
    metadata.create_all(db)
    with db.begin() as connection:
        def empty(table):
            return connection.execute(sa.select(sa.func.count()).select_from(table)).scalar() == 0

        if empty(clinicians):
            seed.load(connection)
        if empty(cultures):       # added after the first databases were filled, so checked by itself
            seed.load_cultures(connection)
        if empty(follow_ups):     # likewise
            seed.load_follow_ups(connection)


# ----------------------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    """A salted scrypt hash, stored as "scrypt$<salt>$<hash>"."""
    salt = secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${_scrypt(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, expected = stored.split("$")
        return scheme == "scrypt" and hmac.compare_digest(
            _scrypt(password, bytes.fromhex(salt)), bytes.fromhex(expected))
    except ValueError:      # not a hash we wrote
        return False


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


_NO_SUCH_USER = hash_password(secrets.token_hex(16))   # checked when the username is unknown


# ----------------------------------------------------------------------------- questions

def sign_in(username: str, password: str, url: str | None = None) -> dict | None:
    """The clinician (id, username, name, role) if the username and password match, else None."""
    with engine(url).connect() as connection:
        row = connection.execute(
            sa.select(clinicians).where(clinicians.c.username == username.strip().lower())
        ).mappings().first()
    # Check a hash even for an unknown username, so both failures take the same time.
    if not verify_password(password, row["password_hash"] if row else _NO_SUCH_USER) or not row:
        return None
    return {key: row[key] for key in ("id", "username", "name", "role")}


def patients_for(clinician_id: int, url: str | None = None) -> list[dict]:
    """The patients assigned to this clinician, in patient-number order."""
    query = (sa.select(patients)
             .join(clinician_patients, clinician_patients.c.patient_id == patients.c.id)
             .where(clinician_patients.c.clinician_id == clinician_id)
             .order_by(patients.c.patient_number))
    with engine(url).connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


def patient_for(clinician_id: int, patient_number: str, url: str | None = None) -> dict | None:
    """One patient, but only if they are assigned to this clinician."""
    query = (sa.select(patients)
             .join(clinician_patients, clinician_patients.c.patient_id == patients.c.id)
             .where(clinician_patients.c.clinician_id == clinician_id,
                    patients.c.patient_number == patient_number))
    with engine(url).connect() as connection:
        row = connection.execute(query).mappings().first()
    return dict(row) if row else None


# ----------------------------------------------------------------------------- cultures

def _cases(connection, clinician_id: int, culture_number: str | None = None) -> list[dict]:
    """Cultures for this clinician's patients, each with its patient, prescription and results."""
    query = (sa.select(cultures, patients.c.patient_number)
             .join(patients, patients.c.id == cultures.c.patient_id)
             .join(clinician_patients, clinician_patients.c.patient_id == patients.c.id)
             .where(clinician_patients.c.clinician_id == clinician_id)
             .order_by(cultures.c.collected_on.desc(), cultures.c.culture_number))
    if culture_number:
        query = query.where(cultures.c.culture_number == culture_number)
    cases = [dict(row) for row in connection.execute(query).mappings()]
    if not cases:
        return []

    results = connection.execute(
        sa.select(susceptibilities).where(susceptibilities.c.culture_id.in_([c["id"] for c in cases]))
        .order_by(susceptibilities.c.antibiotic)).mappings().all()
    # The current prescription is the patient's most recently started one.
    courses = connection.execute(
        sa.select(prescriptions).where(prescriptions.c.patient_id.in_([c["patient_id"] for c in cases]))
        .order_by(prescriptions.c.started_on, prescriptions.c.id)).mappings().all()
    current = {course["patient_id"]: dict(course) for course in courses}
    for case in cases:
        case["susceptibilities"] = {row["antibiotic"]: row["result"] for row in results
                                    if row["culture_id"] == case["id"]}
        case["prescription"] = current.get(case["patient_id"])
    return cases


def cultures_for(clinician_id: int, url: str | None = None) -> list[dict]:
    """Every culture for this clinician's patients, newest first."""
    with engine(url).connect() as connection:
        return _cases(connection, clinician_id)


def culture_for(clinician_id: int, culture_number: str, url: str | None = None) -> dict | None:
    """One culture with its patient's full record, but only for a patient of this clinician."""
    with engine(url).connect() as connection:
        cases = _cases(connection, clinician_id, culture_number)
        if not cases:
            return None
        case = cases[0]
        case["patient"] = dict(connection.execute(
            sa.select(patients).where(patients.c.id == case["patient_id"])).mappings().one())
        if case["reviewed_by"]:
            case["reviewer"] = connection.execute(
                sa.select(clinicians.c.name).where(clinicians.c.id == case["reviewed_by"])).scalar()
    return case


def review_culture(clinician_id: int, culture_number: str, decision: str, notes: str,
                   url: str | None = None) -> bool:
    """Record the clinician's review of a resulted culture. False if it isn't theirs to review."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"Unknown review decision: {decision}")
    with engine(url).begin() as connection:
        cases = _cases(connection, clinician_id, culture_number)
        if not cases or cases[0]["organism"] is None:      # not their patient, or no result yet
            return False
        connection.execute(cultures.update().where(cultures.c.id == cases[0]["id"]).values(
            review_decision=decision, review_notes=notes.strip(), reviewed_by=clinician_id,
            reviewed_at=datetime.datetime.now()))
    return True


def culture_status(case: dict) -> str:
    """Where a culture is in the workflow: Pending, Resistance mismatch, Ready for review, Reviewed."""
    if case["organism"] is None:
        return "Pending"
    if case["review_decision"]:
        return "Reviewed"
    return "Resistance mismatch" if prescription_result(case) == "Resistant" else "Ready for review"


def prescription_result(case: dict) -> str | None:
    """The lab's result for the drug the patient is on, or None if it wasn't tested or isn't back."""
    course = case.get("prescription")
    return case["susceptibilities"].get(course["drug"]) if course else None


# ----------------------------------------------------------------------------- follow-up

def _follow_ups(connection, clinician_id: int, patient_number: str | None = None) -> list[dict]:
    """Follow-ups for this clinician's patients, each with its patient number and prescription."""
    query = (sa.select(follow_ups, patients.c.patient_number)
             .join(patients, patients.c.id == follow_ups.c.patient_id)
             .join(clinician_patients, clinician_patients.c.patient_id == patients.c.id)
             .where(clinician_patients.c.clinician_id == clinician_id)
             .order_by(follow_ups.c.updated_at.desc(), patients.c.patient_number))
    if patient_number:
        query = query.where(patients.c.patient_number == patient_number)
    rows = [dict(row) for row in connection.execute(query).mappings()]
    if rows:
        courses = connection.execute(sa.select(prescriptions).where(
            prescriptions.c.id.in_([row["prescription_id"] for row in rows]))).mappings().all()
        by_id = {course["id"]: dict(course) for course in courses}
        for row in rows:
            row["prescription"] = by_id[row["prescription_id"]]
    return rows


def follow_ups_for(clinician_id: int, url: str | None = None) -> list[dict]:
    """The follow-up queue for this clinician's patients, most recently updated first."""
    with engine(url).connect() as connection:
        return _follow_ups(connection, clinician_id)


def follow_up_for(clinician_id: int, patient_number: str, url: str | None = None) -> dict | None:
    """One patient's follow-up with their record and latest culture, if they are this clinician's."""
    with engine(url).connect() as connection:
        rows = _follow_ups(connection, clinician_id, patient_number)
        if not rows:
            return None
        row = rows[0]
        row["patient"] = dict(connection.execute(
            sa.select(patients).where(patients.c.id == row["patient_id"])).mappings().one())
        cases = [case for case in _cases(connection, clinician_id)        # newest first
                 if case["patient_id"] == row["patient_id"]]
        row["culture"] = cases[0] if cases else None
        if row["decided_by"]:
            row["decider"] = connection.execute(
                sa.select(clinicians.c.name).where(clinicians.c.id == row["decided_by"])).scalar()
    return row


def save_reassessment(clinician_id: int, patient_number: str, decision: str, notes: str,
                      outcome: str, url: str | None = None) -> bool:
    """Record the clinician's reassessment and move the follow-up in the queue. False if not theirs."""
    if decision not in FOLLOW_UP_DECISIONS:
        raise ValueError(f"Unknown clinical decision: {decision}")
    if outcome not in FOLLOW_UP_OUTCOMES:
        raise ValueError(f"Unknown follow-up status: {outcome}")
    now = datetime.datetime.now()
    with engine(url).begin() as connection:
        rows = _follow_ups(connection, clinician_id, patient_number)
        if not rows:
            return False
        connection.execute(follow_ups.update().where(follow_ups.c.id == rows[0]["id"]).values(
            decision=decision, decision_notes=notes.strip(), decided_by=clinician_id, decided_at=now,
            status=FOLLOW_UP_OUTCOMES[outcome], updated_at=now))
    return True
