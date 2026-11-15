"""Sign-in, patient assignment, and turning a patient row into model features."""
import pytest
import sqlalchemy as sa

import data
import db
import seed

FEATURES = ["is_inpatient", "age_min", "gender", "ward_ip", "ward_op", "ward_icu", "adi_score",
            "abx_class_fluoroquinolone_90d", "abx_class_glycopeptide_90d",
            "days_since_last_antibiotic", "prior_org_escherichia", "prior_resist_ciprofloxacin",
            "prior_resist_cefazolin", "n_prior_resistant_drugs", "proc_urethral_catheter",
            "nursing_home"]


@pytest.fixture
def url(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    db.init(url)
    return url


def password_of(username):
    return next(person["password"] for person in seed.CLINICIANS if person["username"] == username)


def test_sign_in_needs_the_right_password(url):
    clinician = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    assert clinician["name"] == "Dr. Sarah Mitchell" and "password_hash" not in clinician
    assert db.sign_in(" SarahMitchell ", password_of("sarahmitchell"), url) == clinician
    assert db.sign_in("sarahmitchell", "wrong", url) is None
    assert db.sign_in("sarahmitchell", password_of("jokafor"), url) is None
    assert db.sign_in("nobody", password_of("sarahmitchell"), url) is None


def test_passwords_are_stored_as_hashes(url):
    with db.engine(url).connect() as connection:
        stored = connection.execute(sa.select(db.clinicians.c.password_hash)).scalars().all()
    assert len(stored) == len(seed.CLINICIANS)
    for person in seed.CLINICIANS:
        assert all(person["password"] not in value for value in stored)
    assert len(set(stored)) == len(stored) and all(v.startswith("scrypt$") for v in stored)


def test_a_clinician_sees_only_their_own_patients(url):
    for person in seed.CLINICIANS:
        clinician = db.sign_in(person["username"], person["password"], url)
        expected = sorted(p["row"]["patient_number"] for p in seed.PATIENTS
                          if person["username"] in p["clinicians"])
        assert [p["patient_number"] for p in db.patients_for(clinician["id"], url)] == expected

    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    okafor = db.sign_in("jokafor", password_of("jokafor"), url)
    assert db.patient_for(mitchell["id"], "PT-00399", url)["patient_number"] == "PT-00399"   # shared
    assert db.patient_for(okafor["id"], "PT-00399", url)["patient_number"] == "PT-00399"
    assert db.patient_for(okafor["id"], "PT-00412", url) is None      # Mitchell's, not Okafor's
    assert db.patient_for(okafor["id"], "PT-99999", url) is None


def test_init_seeds_only_once(url):
    db.init(url)
    with db.engine(url).connect() as connection:
        count = connection.execute(sa.select(sa.func.count()).select_from(db.patients)).scalar()
    assert count == len(seed.PATIENTS)


def test_seed_patients_use_values_the_app_knows():
    for patient in seed.PATIENTS:
        row = patient["row"]
        assert row["age_band"] in data.AGE_BANDS and row["sex"] in data.SEX_CODES
        assert row["care_setting"] in data.SETTINGS
        assert set(patient["clinicians"]) <= {person["username"] for person in seed.CLINICIANS}


def test_patient_features_from_a_database_row(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    record = db.patient_for(mitchell["id"], "PT-00412", url)
    patient = data.patient_features(FEATURES, record)
    assert list(patient) == FEATURES
    assert patient["age_min"] == 75 and patient["gender"] == 0 and patient["nursing_home"] == 1
    assert patient["is_inpatient"] == 1 and patient["ward_icu"] == 1 and patient["ward_op"] == 0
    assert patient["prior_resist_ciprofloxacin"] == 1 and patient["prior_resist_cefazolin"] == 1
    assert patient["n_prior_resistant_drugs"] == 3          # two named + one other
    assert patient["abx_class_fluoroquinolone_90d"] == 1 and patient["abx_class_glycopeptide_90d"] == 1
    assert patient["days_since_last_antibiotic"] == 14
    assert patient["prior_org_escherichia"] == 1 and patient["proc_urethral_catheter"] == 1


def test_lakebase_is_the_default_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ENDPOINT_NAME", raising=False)
    for name, value in {"PGHOST": "ep-test.database.cloud.databricks.com", "PGUSER": "app-client-id",
                        "PGDATABASE": "databricks_postgres", "PGPORT": "5432",
                        "PGSSLMODE": "require"}.items():
        monkeypatch.setenv(name, value)
    url = db.database_url()
    assert db.on_lakebase() and db.database_kind() == "Databricks Lakebase (Postgres)"
    assert db.endpoint_name() == db.DEFAULT_ENDPOINT
    assert (url.drivername, url.host, url.username, url.password) == (
        "postgresql+psycopg", "ep-test.database.cloud.databricks.com", "app-client-id", None)
    assert url.query["sslmode"] == "require" and "search_path=flemingrx" in url.query["options"]


def test_database_url_overrides_lakebase(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/name")
    assert not db.on_lakebase() and db.database_kind() == "Postgres"
    assert db.database_url() == "postgresql+psycopg://u:p@host:5432/name"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///flemingrx.db")
    assert db.database_url() == "sqlite:///flemingrx.db" and db.database_kind() == "SQLite (local file)"


def test_cultures_are_limited_to_the_clinicians_patients(url):
    for person in seed.CLINICIANS:
        clinician = db.sign_in(person["username"], person["password"], url)
        mine = {p["patient_number"] for p in db.patients_for(clinician["id"], url)}
        cases = db.cultures_for(clinician["id"], url)
        assert {case["patient_number"] for case in cases} == mine and len(cases) == len(mine)

    okafor = db.sign_in("jokafor", password_of("jokafor"), url)
    assert db.culture_for(okafor["id"], "CX-2193", url) is None            # PT-00391 is Mitchell's
    assert db.review_culture(okafor["id"], "CX-2193", "Compatible", "", url) is False


def test_culture_status_and_prescription_comparison(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    status = {case["culture_number"]: db.culture_status(case)
              for case in db.cultures_for(mitchell["id"], url)}
    assert status == {"CX-2207": "Pending", "CX-2196": "Ready for review",
                      "CX-2193": "Resistance mismatch", "CX-2199": "Ready for review",
                      "CX-2191": "Ready for review", "CX-2194": "Resistance mismatch",
                      "CX-2188": "Reviewed", "CX-2198": "Ready for review"}

    case = db.culture_for(mitchell["id"], "CX-2193", url)
    assert case["patient"]["patient_number"] == "PT-00391" and case["organism"] == "Escherichia coli"
    assert case["prescription"]["drug"] == "ciprofloxacin"
    assert case["prescription"]["recommended_drug"] == "piperacillin_tazobactam"
    assert db.prescription_result(case) == "Resistant" and len(case["susceptibilities"]) == 8

    pending = db.culture_for(mitchell["id"], "CX-2207", url)
    assert pending["susceptibilities"] == {} and db.prescription_result(pending) is None


def test_reviewing_a_culture(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    assert db.review_culture(mitchell["id"], "CX-2207", "Compatible", "", url) is False   # no result yet
    with pytest.raises(ValueError):
        db.review_culture(mitchell["id"], "CX-2193", "Looks fine", "", url)

    assert db.review_culture(mitchell["id"], "CX-2193", "Reassessment needed", " Switch agent. ", url)
    case = db.culture_for(mitchell["id"], "CX-2193", url)
    assert db.culture_status(case) == "Reviewed" and case["review_notes"] == "Switch agent."
    assert case["reviewer"] == "Dr. Sarah Mitchell" and case["reviewed_at"] is not None


def test_seed_cases_use_known_patients_drugs_and_results():
    numbers = {patient["row"]["patient_number"] for patient in seed.PATIENTS}
    assert {case[0] for case in seed.CASES} == numbers           # one culture per patient
    for case in seed.CASES:
        assert case[2] in seed.REGIMENS and case[3] in seed.REGIMENS
        assert case[9] is None or (len(case[9]) == len(seed.PANEL) and set(case[9]) <= set("SIR"))


def test_follow_up_queue_is_limited_to_the_clinicians_patients(url):
    for person in seed.CLINICIANS:
        clinician = db.sign_in(person["username"], person["password"], url)
        mine = {p["patient_number"] for p in db.patients_for(clinician["id"], url)}
        rows = db.follow_ups_for(clinician["id"], url)
        assert {row["patient_number"] for row in rows} == mine and len(rows) == len(mine)
        assert all(row["status"] in db.FOLLOW_UP_STATUSES and row["reason"] in db.FOLLOW_UP_REASONS
                   for row in rows)

    okafor = db.sign_in("jokafor", password_of("jokafor"), url)
    assert db.follow_up_for(okafor["id"], "PT-00391", url) is None         # Mitchell's patient
    assert db.save_reassessment(okafor["id"], "PT-00391", "Modify treatment", "",
                                "Resolved", url) is False


def test_follow_up_carries_the_prescription_and_latest_culture(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    row = db.follow_up_for(mitchell["id"], "PT-00391", url)
    assert row["reason"] == "Culture resistance mismatch" and row["status"] == "Needs reassessment"
    assert row["prescription"]["drug"] == "ciprofloxacin" and row["patient"]["age_band"] == "75-84"
    assert row["culture"]["culture_number"] == "CX-2193"
    assert db.prescription_result({**row["culture"], "prescription": row["prescription"]}) == "Resistant"

    statuses = [r["status"] for r in db.follow_ups_for(mitchell["id"], url)]
    assert {s: statuses.count(s) for s in db.FOLLOW_UP_STATUSES} == {
        "Needs reassessment": 4, "Awaiting follow-up": 3, "Resolved": 1}


def test_saving_a_reassessment_moves_the_follow_up(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    with pytest.raises(ValueError):
        db.save_reassessment(mitchell["id"], "PT-00391", "Do nothing", "", "Resolved", url)
    with pytest.raises(ValueError):
        db.save_reassessment(mitchell["id"], "PT-00391", "Modify treatment", "", "Done", url)

    assert db.save_reassessment(mitchell["id"], "PT-00391", "Modify treatment", " Switched. ",
                                "Further monitoring required", url)
    row = db.follow_up_for(mitchell["id"], "PT-00391", url)
    assert (row["status"], row["decision"], row["decision_notes"]) == (
        "Awaiting follow-up", "Modify treatment", "Switched.")
    assert row["decider"] == "Dr. Sarah Mitchell"
    assert db.follow_ups_for(mitchell["id"], url)[0]["patient_number"] == "PT-00391"   # newest first


def test_seed_follow_ups_use_known_patients_and_values():
    numbers = {patient["row"]["patient_number"] for patient in seed.PATIENTS}
    assert {row[0] for row in seed.FOLLOW_UPS} == numbers          # one follow-up per patient
    for row in seed.FOLLOW_UPS:
        assert row[1] in db.FOLLOW_UP_REASONS and row[5] in db.FOLLOW_UP_STATUSES
        assert row[10] is None or row[10][0] in db.FOLLOW_UP_DECISIONS
