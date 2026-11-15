"""Fabricated demo data: three clinicians, their patients, and each patient's prescription, urine
culture and follow-up.

NONE OF THIS IS REAL. The clinicians and patients are made up for the hackathon demo, and the
passwords below are demo passwords, here so anyone running the demo can sign in. Do not reuse
them anywhere, and replace this file before the app goes near real data.

db.init() loads this once, into an empty database. To reload it after editing, empty the three
tables (in the Lakebase SQL editor: TRUNCATE flemingrx.clinicians, flemingrx.patients CASCADE)
and start the app again.
"""

import datetime

import db
import policy

CLINICIANS = [
    {"username": "sarahmitchell", "name": "Dr. Sarah Mitchell", "role": "Infectious Disease",
     "password": "password"},
    {"username": "jokafor", "name": "Dr. James Okafor", "role": "Urology",
     "password": "fleming-demo-2"},
    {"username": "praman", "name": "Dr. Priya Raman", "role": "Emergency Medicine",
     "password": "fleming-demo-3"},
]


VITALS = ("temperature_f", "heart_rate", "respiratory_rate", "systolic_bp")


def _patient(number, clinicians, age_band, sex, care_setting, nursing_home=False, resistant_to=(),
             other_resistant_count=0, recent_antibiotic_classes=(), days_since_last_antibiotic=366,
             prior_organisms=(), recent_procedures=(), symptoms=("Dysuria", "Urinary frequency"),
             allergies=(), egfr=88, pregnancy=None, vitals=(98.4, 78, 16, 122)):
    # vitals: (temperature in Fahrenheit, heart rate, breathing rate, systolic blood pressure);
    # None for one that wasn't recorded
    if pregnancy is None:     # only asked of women of childbearing age
        of_age = age_band in ("18-24", "25-34", "35-44", "45-54")
        pregnancy = "Not pregnant" if sex == "Female" and of_age else "Not applicable"
    return {
        "clinicians": clinicians,
        "row": {
            "patient_number": number, "age_band": age_band, "sex": sex, "care_setting": care_setting,
            "nursing_home": nursing_home, "resistant_to": list(resistant_to),
            "other_resistant_count": other_resistant_count,
            "recent_antibiotic_classes": list(recent_antibiotic_classes),
            "days_since_last_antibiotic": days_since_last_antibiotic,
            "prior_organisms": list(prior_organisms), "recent_procedures": list(recent_procedures),
            "symptoms": list(symptoms), "allergies": list(allergies), "egfr": egfr,
            "pregnancy": pregnancy,
            **dict(zip(VITALS, vitals)),
        },
    }


PATIENTS = [
    _patient("PT-00377", ["sarahmitchell"], "25-34", "Female", "Outpatient clinic"),
    _patient("PT-00389", ["sarahmitchell", "praman"], "65-74", "Female", "Emergency department",
             recent_antibiotic_classes=["beta_lactam"], days_since_last_antibiotic=14,
             prior_organisms=["escherichia"]),
    _patient("PT-00391", ["sarahmitchell"], "75-84", "Male", "Hospital ward",
             resistant_to=["ciprofloxacin"], recent_antibiotic_classes=["fluoroquinolone"],
             days_since_last_antibiotic=60, prior_organisms=["escherichia", "enterococcus"],
             recent_procedures=["urethral_catheter"], symptoms=["Dysuria", "Fever"], egfr=61,
             vitals=(100.8, 88, 18, 126)),
    _patient("PT-00399", ["sarahmitchell", "jokafor"], "35-44", "Female", "Outpatient clinic",
             resistant_to=["trimethoprim_sulfamethoxazole"],
             recent_antibiotic_classes=["sulfonamide"], days_since_last_antibiotic=60,
             prior_organisms=["escherichia"]),
    _patient("PT-00401", ["sarahmitchell"], "85-89", "Female", "Hospital ward", nursing_home=True,
             resistant_to=["nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin",
                           "ciprofloxacin", "ceftriaxone"],
             other_resistant_count=3, recent_antibiotic_classes=["beta_lactam", "fluoroquinolone"],
             days_since_last_antibiotic=14, prior_organisms=["klebsiella", "escherichia"],
             recent_procedures=["urethral_catheter"], symptoms=["Fever", "Suprapubic pain"],
             egfr=38, vitals=(101.2, 96, 20, 112)),
    _patient("PT-00408", ["sarahmitchell", "jokafor"], "55-64", "Male", "Outpatient clinic",
             recent_antibiotic_classes=["nitrofuran"], days_since_last_antibiotic=60,
             prior_organisms=["proteus"], recent_procedures=["surgical_procedure"]),
    _patient("PT-00410", ["sarahmitchell"], "45-54", "Female", "Outpatient clinic",
             days_since_last_antibiotic=180),
    _patient("PT-00412", ["sarahmitchell", "praman"], "75-84", "Female", "Intensive care unit",
             nursing_home=True, resistant_to=["ciprofloxacin", "cefazolin"], other_resistant_count=1,
             recent_antibiotic_classes=["fluoroquinolone", "glycopeptide"],
             days_since_last_antibiotic=14, prior_organisms=["escherichia", "pseudomonas"],
             recent_procedures=["urethral_catheter", "cvc"], symptoms=["Fever", "Flank pain"],
             allergies=["sulfonamide"], egfr=52, vitals=(102.1, 112, 24, 88)),
    _patient("PT-00403", ["jokafor"], "65-74", "Male", "Hospital ward",
             resistant_to=["cefazolin"], recent_antibiotic_classes=["beta_lactam"],
             days_since_last_antibiotic=14, prior_organisms=["enterococcus"],
             recent_procedures=["urethral_catheter", "surgical_procedure"]),
    _patient("PT-00409", ["jokafor"], "18-24", "Female", "Outpatient clinic"),
    _patient("PT-00415", ["jokafor", "praman"], "90 or older", "Female", "Emergency department",
             nursing_home=True, resistant_to=["nitrofurantoin"],
             recent_antibiotic_classes=["nitrofuran"], days_since_last_antibiotic=60,
             prior_organisms=["klebsiella"]),
    _patient("PT-00418", ["praman"], "25-34", "Male", "Emergency department"),
    _patient("PT-00421", ["praman"], "55-64", "Female", "Emergency department",
             resistant_to=["trimethoprim_sulfamethoxazole", "ciprofloxacin"],
             recent_antibiotic_classes=["fluoroquinolone"], days_since_last_antibiotic=14,
             prior_organisms=["escherichia"], recent_procedures=["dialysis"], egfr=12),

    # Four patients with no prescription yet, so the accept / override step can be shown. Between
    # them they trip each of the pre-checks.
    _patient("PT-00424", ["sarahmitchell"], "25-34", "Female", "Outpatient clinic",
             symptoms=["Dysuria", "Urinary frequency", "Urgency"]),
    _patient("PT-00427", ["sarahmitchell"], "75-84", "Female", "Outpatient clinic",
             nursing_home=True, prior_organisms=["escherichia"], symptoms=[], egfr=64,
             vitals=(None, None, None, None)),
    _patient("PT-00430", ["sarahmitchell", "jokafor"], "25-34", "Female", "Outpatient clinic",
             resistant_to=["nitrofurantoin", "cefazolin"], prior_organisms=["escherichia"],
             symptoms=["Dysuria", "Urgency"], allergies=["sulfonamide"], pregnancy="Pregnant",
             vitals=(100.9, 86, 16, 118)),
    _patient("PT-00433", ["sarahmitchell", "praman"], "65-74", "Male", "Hospital ward",
             resistant_to=["ciprofloxacin", "ceftriaxone"],
             recent_antibiotic_classes=["fluoroquinolone"], days_since_last_antibiotic=14,
             prior_organisms=["klebsiella"], recent_procedures=["urethral_catheter"],
             symptoms=["Fever", "Suprapubic pain"], allergies=["penicillin"], egfr=24,
             vitals=(101.5, 104, 22, 96)),
]


# The antibiotics on the lab's panel, in the order of the S/R letters in CASES below.
PANEL = ["ampicillin", "nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin",
         "ciprofloxacin", "ceftriaxone", "piperacillin_tazobactam", "meropenem"]

# One prescription and one culture for each of the first 13 patients:
# (patient, culture number, model's recommendation at a 90% bar, drug prescribed, prescribed by,
#  days since the culture was collected, specimen, organism, colony count, panel results, review)
# The recommendations are what the model really returns for each fabricated patient.
# Organism None means the lab has not reported yet. `review` is (decision, notes, reviewer) or None.
CASES = [
    ("PT-00377", "CX-2207", "cefazolin", "nitrofurantoin", "sarahmitchell", 1,
     "Urine, clean catch", None, None, None, None),
    ("PT-00389", "CX-2196", "piperacillin_tazobactam", "ceftriaxone", "praman", 3,
     "Urine, clean catch", "Escherichia coli", 100000, "RSSSSSSS", None),
    ("PT-00391", "CX-2193", "piperacillin_tazobactam", "ciprofloxacin", "sarahmitchell", 3,
     "Urine, catheter", "Escherichia coli", 100000, "RSRSRSSS", None),
    ("PT-00399", "CX-2199", "nitrofurantoin", "nitrofurantoin", "sarahmitchell", 2,
     "Urine, clean catch", "Escherichia coli", 100000, "RSRSSSSS", None),
    ("PT-00401", "CX-2191", "piperacillin_tazobactam", "piperacillin_tazobactam", "sarahmitchell", 4,
     "Urine, catheter", "Klebsiella pneumoniae", 100000, "RRRRRRSS", None),
    ("PT-00408", "CX-2194", "piperacillin_tazobactam", "trimethoprim_sulfamethoxazole", "jokafor", 3,
     "Urine, clean catch", "Proteus mirabilis", 50000, "SRRSSSSS", None),
    ("PT-00410", "CX-2188", "ceftriaxone", "nitrofurantoin", "sarahmitchell", 5,
     "Urine, clean catch", "Escherichia coli", 100000, "SSSSSSSS",
     ("Compatible", "Susceptible to nitrofurantoin. Continue the 5-day course.", "sarahmitchell")),
    ("PT-00412", "CX-2198", "piperacillin_tazobactam", "piperacillin_tazobactam", "sarahmitchell", 2,
     "Urine, catheter", "Escherichia coli", 100000, "RSSRRSSS", None),
    ("PT-00403", "CX-2190", "piperacillin_tazobactam", "piperacillin_tazobactam", "jokafor", 4,
     "Urine, catheter", "Enterococcus faecalis", 50000, "SSRRRRSS", None),
    ("PT-00409", "CX-2205", "cefazolin", "nitrofurantoin", "jokafor", 1,
     "Urine, clean catch", None, None, None, None),
    ("PT-00415", "CX-2201", "piperacillin_tazobactam", "ceftriaxone", "praman", 2,
     "Urine, clean catch", "Klebsiella pneumoniae", 100000, "RRSRSRSS", None),
    ("PT-00418", "CX-2187", "piperacillin_tazobactam", "ciprofloxacin", "praman", 5,
     "Urine, clean catch", "Escherichia coli", 100000, "RSSSSSSS",
     ("Compatible", "Susceptible to ciprofloxacin. No change.", "praman")),
    ("PT-00421", "CX-2200", "piperacillin_tazobactam", "ceftriaxone", "praman", 2,
     "Urine, clean catch", "Escherichia coli", 100000, "RSRSRSSS", None),
]

_RESULT = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant"}


# Why the clinician departed from the model for the demo's overridden prescriptions.
OVERRIDES = {
    "PT-00377": ("Clinician preference", "Uncomplicated cystitis; first-line oral agent preferred."),
    "PT-00389": ("Patient-specific considerations", "Stable for a once-daily agent."),
    "PT-00391": ("Clinician preference", "Oral step-down planned."),
    "PT-00408": ("Patient-specific considerations", "Outpatient; oral agent needed."),
    "PT-00410": ("Clinician preference", "Uncomplicated cystitis; first-line oral agent preferred."),
    "PT-00409": ("Clinician preference", "Uncomplicated cystitis; first-line oral agent preferred."),
    "PT-00415": ("Patient-specific considerations", "Once-daily dosing in the emergency department."),
    "PT-00418": ("Clinician preference", "Oral agent for discharge."),
    "PT-00421": ("Kidney function", "On dialysis; agent chosen with the renal team."),
}


def load(connection) -> None:
    """Insert the clinicians and patients that are not in the database yet, and who looks after whom.

    Run on every start: on an empty database it adds everything, and after CLINICIANS or PATIENTS
    gains an entry it adds just that one.
    """
    clinician_ids = dict(connection.execute(
        db.sa.select(db.clinicians.c.username, db.clinicians.c.id)).all())
    for person in CLINICIANS:
        if person["username"] not in clinician_ids:
            result = connection.execute(db.clinicians.insert().values(
                username=person["username"], name=person["name"], role=person["role"],
                password_hash=db.hash_password(person["password"])))
            clinician_ids[person["username"]] = result.inserted_primary_key[0]
    existing = set(connection.execute(db.sa.select(db.patients.c.patient_number)).scalars())
    for patient in PATIENTS:
        if patient["row"]["patient_number"] in existing:
            continue
        result = connection.execute(db.patients.insert().values(**patient["row"]))
        patient_id = result.inserted_primary_key[0]
        for username in patient["clinicians"]:
            connection.execute(db.clinician_patients.insert().values(
                clinician_id=clinician_ids[username], patient_id=patient_id))


def fill_new_columns(connection) -> None:
    """Give demo rows that predate a column its demo value. Only ever fills empty values."""
    new = ("symptoms", "allergies", "egfr", "pregnancy")
    for patient in PATIENTS:
        row = patient["row"]
        connection.execute(db.patients.update()
                           .where(db.patients.c.patient_number == row["patient_number"],
                                  db.patients.c.symptoms.is_(None))
                           .values({name: row[name] for name in new}))
        # Vital signs came later still. All four empty means "not filled yet" for a demo patient,
        # unless the demo patient really has none recorded, in which case there is nothing to set.
        if any(row[name] is not None for name in VITALS):
            connection.execute(db.patients.update()
                               .where(db.patients.c.patient_number == row["patient_number"],
                                      *[db.patients.c[name].is_(None) for name in VITALS])
                               .values({name: row[name] for name in VITALS}))
    # Demo prescriptions from before decisions were recorded: work the decision out from the drugs.
    undecided = connection.execute(
        db.sa.select(db.prescriptions.c.id, db.prescriptions.c.drug,
                     db.prescriptions.c.recommended_drug, db.patients.c.patient_number)
        .join(db.patients, db.patients.c.id == db.prescriptions.c.patient_id)
        .where(db.prescriptions.c.decision.is_(None))).all()
    for course_id, drug, recommended, patient_number in undecided:
        accepted = drug == recommended
        reason, notes = (None, None) if accepted else OVERRIDES.get(patient_number, ("Other", None))
        connection.execute(db.prescriptions.update().where(db.prescriptions.c.id == course_id).values(
            decision="Accepted" if accepted else "Overridden", override_reason=reason,
            override_notes=notes))

    # Prescriptions from before assessments were logged: give each a log row saying what was
    # recommended and decided. What the model predicted at the time was never stored, so
    # `chances`, `cleared_bar` and `prechecks` stay empty for these.
    unlogged = connection.execute(
        db.sa.select(db.prescriptions).where(db.prescriptions.c.assessment_id.is_(None))
    ).mappings().all()
    for course in unlogged:
        when = min(datetime.datetime.combine(course["started_on"], datetime.time(9, 0)),
                   datetime.datetime.now())
        result = connection.execute(db.assessments.insert().values(
            patient_id=course["patient_id"], clinician_id=course["prescribed_by"], assessed_at=when,
            target=0.9, recommended_drug=course["recommended_drug"], outcome=course["decision"],
            outcome_reason=course["override_reason"], outcome_notes=course["override_notes"],
            decided_at=when))
        connection.execute(db.prescriptions.update().where(db.prescriptions.c.id == course["id"])
                           .values(assessment_id=result.inserted_primary_key[0]))


def load_cultures(connection) -> None:
    """Insert a prescription and a culture (with its panel results) for each patient in CASES.

    Dates are counted back from today, so the demo always looks current.
    """
    today = datetime.date.today()
    patient_ids = dict(connection.execute(
        db.sa.select(db.patients.c.patient_number, db.patients.c.id)).all())
    clinician_ids = dict(connection.execute(
        db.sa.select(db.clinicians.c.username, db.clinicians.c.id)).all())
    for (patient, number, recommended, drug, prescriber, days_ago, specimen, organism, colonies,
         panel, review) in CASES:
        if patient not in patient_ids:
            continue
        collected = today - datetime.timedelta(days=days_ago)
        dose, frequency, duration = policy.USUAL_REGIMENS[drug]
        connection.execute(db.prescriptions.insert().values(
            patient_id=patient_ids[patient], prescribed_by=clinician_ids[prescriber],
            recommended_drug=recommended, drug=drug, dose=dose, frequency=frequency,
            duration_days=duration, started_on=collected))
        decision, notes, reviewer = review or (None, None, None)
        result = connection.execute(db.cultures.insert().values(
            culture_number=number, patient_id=patient_ids[patient], specimen=specimen,
            collected_on=collected,
            resulted_on=collected + datetime.timedelta(days=2) if organism else None,
            organism=organism, colony_count=colonies, review_decision=decision, review_notes=notes,
            reviewed_by=clinician_ids.get(reviewer),
            reviewed_at=min(datetime.datetime.combine(today, datetime.time(9, 0)),
                            datetime.datetime.now()) if review else None))
        for antibiotic, letter in zip(PANEL, panel or ""):
            connection.execute(db.susceptibilities.insert().values(
                culture_id=result.inserted_primary_key[0], antibiotic=antibiotic,
                result=_RESULT[letter]))


# One follow-up for each patient who has a prescription:
# (patient, reason, why, where that came from, days since it was reported, queue status,
#  symptom progression, treatment adherence, adverse reaction, additional notes, reassessment)
# None means not documented yet. `reassessment` is (decision, notes, clinician) or None.
TAKING = "Taking medication as prescribed"
FOLLOW_UPS = [
    ("PT-00391", "Culture resistance mismatch",
     "Culture results show resistance to the prescribed antibiotic (ciprofloxacin).",
     "Urine culture CX-2193", 1, "Needs reassessment",
     "Not improving", TAKING, None, "Persistent dysuria and low-grade fever.", None),
    ("PT-00408", "Culture resistance mismatch",
     "Culture results show resistance to the prescribed antibiotic (trimethoprim/sulfamethoxazole).",
     "Urine culture CX-2194", 1, "Needs reassessment",
     "Unchanged", TAKING, None, None, None),
    ("PT-00415", "Culture resistance mismatch",
     "Culture results show resistance to the prescribed antibiotic (ceftriaxone).",
     "Urine culture CX-2201", 0, "Needs reassessment",
     "Not improving", TAKING, None, "Remains febrile in the emergency department.", None),
    ("PT-00401", "Adverse drug reaction",
     "Patient experienced a documented adverse reaction after starting treatment.",
     "Reported during a subsequent clinical encounter", 1, "Needs reassessment",
     "Improving", TAKING, "Maculopapular rash on the trunk, day 2 of piperacillin/tazobactam",
     "No airway or mucosal involvement documented.", None),
    ("PT-00412", "Symptoms worsening or not improving",
     "Patient has documented persistent urinary symptoms and fever following initiation of "
     "antibiotic treatment.",
     "Reported during a subsequent clinical encounter", 0, "Needs reassessment",
     "Worsening", TAKING, None, "Temperature 38.6 C at 48 hours.", None),
    ("PT-00403", "Other clinical concern",
     "Urinary catheter has been in place throughout treatment and is due for review.",
     "Reported during a subsequent clinical encounter", 1, "Needs reassessment",
     "Improving", TAKING, None, "Catheter change not yet documented.", None),
    ("PT-00377", "Follow-up pending",
     "Treatment started; no follow-up information has been documented yet.",
     "Original assessment", 1, "Awaiting follow-up", None, None, None, None, None),
    ("PT-00389", "Follow-up pending",
     "Treatment started; no follow-up information has been documented yet.",
     "Original assessment", 2, "Awaiting follow-up", None, None, None, None, None),
    ("PT-00399", "Follow-up pending",
     "Treatment started; no follow-up information has been documented yet.",
     "Original assessment", 2, "Awaiting follow-up", None, None, None, None, None),
    ("PT-00409", "Follow-up pending",
     "Treatment started; no follow-up information has been documented yet.",
     "Original assessment", 1, "Awaiting follow-up", None, None, None, None, None),
    ("PT-00421", "Follow-up pending",
     "Treatment started; no follow-up information has been documented yet.",
     "Original assessment", 2, "Awaiting follow-up", None, None, None, None, None),
    ("PT-00410", "Treatment completion",
     "Patient has completed treatment and requires an outcome assessment.",
     "Reported during a subsequent clinical encounter", 0, "Resolved",
     "Resolved", "Completed the full course", None, "No urinary symptoms at review.",
     ("Continue current treatment", "Course completed; symptoms resolved. No further action.",
      "sarahmitchell")),
    ("PT-00418", "Adverse drug reaction",
     "Patient experienced a documented adverse reaction after starting treatment.",
     "Reported during a subsequent clinical encounter", 2, "Resolved",
     "Improving", TAKING, "Mild nausea with ciprofloxacin, settled when taken with food",
     None,
     ("Continue current treatment", "Nausea settled; continue the course.", "praman")),
]


def load_follow_ups(connection) -> None:
    """Insert the follow-up for each patient in FOLLOW_UPS, tied to their prescription."""
    today = datetime.date.today()
    patient_ids = dict(connection.execute(
        db.sa.select(db.patients.c.patient_number, db.patients.c.id)).all())
    clinician_ids = dict(connection.execute(
        db.sa.select(db.clinicians.c.username, db.clinicians.c.id)).all())
    course_ids = dict(connection.execute(
        db.sa.select(db.prescriptions.c.patient_id, db.prescriptions.c.id)
        .order_by(db.prescriptions.c.started_on, db.prescriptions.c.id)).all())   # latest wins
    for (patient, reason, detail, source, days_ago, status, symptoms, adherence, reaction, notes,
         reassessment) in FOLLOW_UPS:
        patient_id = patient_ids.get(patient)
        if patient_id not in course_ids:
            continue
        reported = today - datetime.timedelta(days=days_ago)
        # 09:30 on the day it was reported, but never later than now
        updated = min(datetime.datetime.combine(reported, datetime.time(9, 30)),
                      datetime.datetime.now())
        decision, decision_notes, decider = reassessment or (None, None, None)
        connection.execute(db.follow_ups.insert().values(
            patient_id=patient_id, prescription_id=course_ids[patient_id], reason=reason,
            reason_detail=detail, source=source, reported_on=reported, status=status,
            updated_at=updated, symptom_progression=symptoms, treatment_adherence=adherence,
            adverse_reaction=reaction, additional_notes=notes, decision=decision,
            decision_notes=decision_notes, decided_by=clinician_ids.get(decider),
            decided_at=updated if reassessment else None))
