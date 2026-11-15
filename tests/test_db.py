"""Sign-in, patient assignment, and turning a patient row into model features."""
import pytest
import sqlalchemy as sa

import data
import db
import policy
import prechecks
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
        mine &= {case[0] for case in seed.CASES}             # the newest patients have no culture yet
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
    assert {case[0] for case in seed.CASES} <= numbers and len(seed.CASES) == 13
    for case in seed.CASES:
        assert case[2] in policy.USUAL_REGIMENS and case[3] in policy.USUAL_REGIMENS
        assert case[9] is None or (len(case[9]) == len(seed.PANEL) and set(case[9]) <= set("SIR"))


def test_follow_up_queue_is_limited_to_the_clinicians_patients(url):
    for person in seed.CLINICIANS:
        clinician = db.sign_in(person["username"], person["password"], url)
        mine = {p["patient_number"] for p in db.patients_for(clinician["id"], url)}
        mine &= {row[0] for row in seed.FOLLOW_UPS}          # only patients with a prescription
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
    assert {row[0] for row in seed.FOLLOW_UPS} == {case[0] for case in seed.CASES} <= numbers
    for row in seed.FOLLOW_UPS:
        assert row[1] in db.FOLLOW_UP_REASONS and row[5] in db.FOLLOW_UP_STATUSES
        assert row[10] is None or row[10][0] in db.FOLLOW_UP_DECISIONS


def test_analytics_counts_come_from_the_database(url):
    import datetime

    import analytics

    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    rows = (db.prescriptions_for(mitchell["id"], url), db.cultures_for(mitchell["id"], url),
            db.follow_ups_for(mitchell["id"], url))
    stats = analytics.summarise(*rows)
    assert (stats["assessments"], stats["accepted"], stats["changed"]) == (8, 3, 5)
    assert sorted(stats["changed_patients"]) == ["PT-00377", "PT-00389", "PT-00391", "PT-00408",
                                                 "PT-00410"]
    assert stats["by_drug"][0] == ("nitrofurantoin", 3) and sum(n for _, n in stats["by_drug"]) == 8
    assert sorted(stats["patients_by_drug"]["nitrofurantoin"]) == ["PT-00377", "PT-00399", "PT-00410"]
    assert (stats["cultures"], stats["resulted"], stats["mismatches"], stats["reviewed"]) == (8, 7, 2, 1)
    assert dict((a, (r, t)) for a, r, t in stats["resistance"])["ciprofloxacin"] == (3, 7)
    assert stats["organisms"][0] == ("Escherichia coli", 5)
    assert (stats["follow_ups"], stats["documented"], stats["adverse"]) == (8, 5, 1)
    assert dict(stats["outcomes"]) == {"Resolved": 1, "Improving": 1, "Unchanged": 1,
                                       "Not improving": 1, "Worsening": 1}
    assert (stats["unresolved"], stats["needs_reassessment"]) == (3, 4)

    # The reporting period drops anything older: only PT-00377's culture was collected yesterday.
    recent = analytics.summarise(*rows, since=datetime.date.today() - datetime.timedelta(days=1))
    assert (recent["assessments"], recent["cultures"], recent["resulted"]) == (1, 1, 0)
    nothing = analytics.summarise(*rows, since=datetime.date.today() + datetime.timedelta(days=1))
    assert nothing["assessments"] == 0 and analytics.percent(0, 0) == "–"


def test_accepting_and_overriding_a_recommendation(url):
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    okafor = db.sign_in("jokafor", password_of("jokafor"), url)
    regimen = policy.USUAL_REGIMENS["cefazolin"]
    assert db.prescription_for(mitchell["id"], "PT-00424", url) is None

    assert db.save_prescription(okafor["id"], "PT-00424", "cefazolin", "cefazolin", regimen,
                                url=url) is False                       # not Okafor's patient
    assert db.save_prescription(mitchell["id"], "PT-00424", "cefazolin", "cefazolin", regimen, url=url)
    course = db.prescription_for(mitchell["id"], "PT-00424", url)
    assert (course["decision"], course["drug"], course["dose"], course["duration_days"]) == (
        "Accepted", "cefazolin", "1 g IV", 7)
    assert course["override_reason"] is None and course["prescriber"] == "Dr. Sarah Mitchell"
    assert db.save_prescription(mitchell["id"], "PT-00424", "cefazolin", "cefazolin", regimen,
                                url=url) is False                       # already has one

    with pytest.raises(ValueError):                                     # an override needs a reason
        db.save_prescription(mitchell["id"], "PT-00427", "cefazolin", "nitrofurantoin",
                             policy.USUAL_REGIMENS["nitrofurantoin"], url=url)
    assert db.save_prescription(mitchell["id"], "PT-00427", "cefazolin", "nitrofurantoin",
                                policy.USUAL_REGIMENS["nitrofurantoin"], "Clinician preference",
                                " Oral agent. ", url)
    course = db.prescription_for(mitchell["id"], "PT-00427", url)
    assert (course["decision"], course["override_reason"], course["override_notes"]) == (
        "Overridden", "Clinician preference", "Oral agent.")


def test_demo_prescriptions_have_a_decision_and_overrides_a_reason(url):
    with db.engine(url).connect() as connection:
        rows = connection.execute(sa.select(db.prescriptions)).mappings().all()
    assert len(rows) == 13
    for row in rows:
        accepted = row["drug"] == row["recommended_drug"]
        assert row["decision"] == ("Accepted" if accepted else "Overridden")
        assert (row["override_reason"] is None) == accepted
        assert accepted or row["override_reason"] in db.OVERRIDE_REASONS


def test_an_older_database_gains_the_new_columns_and_patients(tmp_path):
    """A database made before the pre-check and decision columns existed is brought up to date."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    db.init(url)
    with db.engine(url).begin() as connection:
        for table, column in [("patients", "symptoms"), ("patients", "allergies"), ("patients", "egfr"),
                              ("patients", "pregnancy"), ("prescriptions", "decision"),
                              ("prescriptions", "override_reason"), ("prescriptions", "override_notes")]:
            connection.execute(sa.text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        connection.execute(sa.text("DELETE FROM clinician_patients WHERE patient_id IN "
                                   "(SELECT id FROM patients WHERE patient_number = 'PT-00433')"))
        connection.execute(sa.text("DELETE FROM patients WHERE patient_number = 'PT-00433'"))
    db.engine(url).dispose()

    db.init(url)
    mitchell = db.sign_in("sarahmitchell", password_of("sarahmitchell"), url)
    old = db.patient_for(mitchell["id"], "PT-00412", url)
    assert (old["allergies"], old["egfr"], old["pregnancy"]) == (["sulfonamide"], 52, "Not applicable")
    assert db.patient_for(mitchell["id"], "PT-00433", url)["egfr"] == 24            # added back
    assert db.prescription_for(mitchell["id"], "PT-00391", url)["decision"] == "Overridden"
    assert len(db.patients_for(mitchell["id"], url)) == 12


def test_prechecks_flag_what_gets_in_the_way():
    base = {"symptoms": ["Dysuria"], "allergies": [], "egfr": 90, "pregnancy": "Not pregnant"}
    levels = lambda record, drug: [c["level"] for c in prechecks.run(record, drug)]   # noqa: E731
    assert levels(base, "nitrofurantoin") == ["ok", "ok", "ok", "ok"]

    # No symptoms: usually don't treat, unless pregnant.
    assert prechecks.symptoms({**base, "symptoms": []})["level"] == "stop"
    assert prechecks.symptoms({**base, "symptoms": [], "pregnancy": "Pregnant"})["level"] == "caution"

    allergic = {**base, "allergies": ["penicillin"]}
    assert prechecks.allergies(allergic, "piperacillin_tazobactam")["level"] == "stop"
    assert prechecks.allergies(allergic, "ceftriaxone")["level"] == "caution"       # related drug
    assert prechecks.allergies(allergic, "nitrofurantoin")["level"] == "ok"

    assert prechecks.kidney({**base, "egfr": 24}, "nitrofurantoin")["level"] == "stop"
    assert prechecks.kidney({**base, "egfr": 24}, "ceftriaxone")["level"] == "caution"
    assert prechecks.kidney({**base, "egfr": 45}, "ceftriaxone")["level"] == "caution"
    assert prechecks.kidney({**base, "egfr": None}, "ceftriaxone")["level"] == "unknown"

    pregnant = {**base, "pregnancy": "Pregnant"}
    assert prechecks.pregnancy(pregnant, "ciprofloxacin")["level"] == "stop"
    assert prechecks.pregnancy(pregnant, "cefazolin")["level"] == "caution"
    assert prechecks.pregnancy({**base, "pregnancy": None}, "cefazolin")["level"] == "unknown"

    assert prechecks.worst(prechecks.run(pregnant, "ciprofloxacin")) == "stop"
    assert set(prechecks.DRUG_GROUP) == set(policy.DRUG_LABELS) == set(policy.USUAL_REGIMENS)
    for patient in seed.PATIENTS:                       # the demo data only uses known values
        assert set(patient["row"]["allergies"]) <= set(prechecks.ALLERGY_LABELS)
        assert patient["row"]["pregnancy"] in ("Pregnant", "Not pregnant", "Not applicable")
