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
