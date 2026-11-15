"""Fabricated demo data: three clinicians, their patients, and each patient's prescription and
urine culture.

NONE OF THIS IS REAL. The clinicians and patients are made up for the hackathon demo, and the
passwords below are demo passwords, here so anyone running the demo can sign in. Do not reuse
them anywhere, and replace this file before the app goes near real data.

db.init() loads this once, into an empty database. To reload it after editing, empty the three
tables (in the Lakebase SQL editor: TRUNCATE flemingrx.clinicians, flemingrx.patients CASCADE)
and start the app again.
"""

import datetime

import db

CLINICIANS = [
    {"username": "sarahmitchell", "name": "Dr. Sarah Mitchell", "role": "Infectious Disease",
     "password": "password"},
    {"username": "jokafor", "name": "Dr. James Okafor", "role": "Urology",
     "password": "fleming-demo-2"},
    {"username": "praman", "name": "Dr. Priya Raman", "role": "Emergency Medicine",
     "password": "fleming-demo-3"},
]


def _patient(number, clinicians, age_band, sex, care_setting, nursing_home=False, resistant_to=(),
             other_resistant_count=0, recent_antibiotic_classes=(), days_since_last_antibiotic=366,
             prior_organisms=(), recent_procedures=()):
    return {
        "clinicians": clinicians,
        "row": {
            "patient_number": number, "age_band": age_band, "sex": sex, "care_setting": care_setting,
            "nursing_home": nursing_home, "resistant_to": list(resistant_to),
            "other_resistant_count": other_resistant_count,
            "recent_antibiotic_classes": list(recent_antibiotic_classes),
            "days_since_last_antibiotic": days_since_last_antibiotic,
            "prior_organisms": list(prior_organisms), "recent_procedures": list(recent_procedures),
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
             recent_procedures=["urethral_catheter"]),
    _patient("PT-00399", ["sarahmitchell", "jokafor"], "35-44", "Female", "Outpatient clinic",
             resistant_to=["trimethoprim_sulfamethoxazole"],
             recent_antibiotic_classes=["sulfonamide"], days_since_last_antibiotic=60,
             prior_organisms=["escherichia"]),
    _patient("PT-00401", ["sarahmitchell"], "85-89", "Female", "Hospital ward", nursing_home=True,
             resistant_to=["nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin",
                           "ciprofloxacin", "ceftriaxone"],
             other_resistant_count=3, recent_antibiotic_classes=["beta_lactam", "fluoroquinolone"],
             days_since_last_antibiotic=14, prior_organisms=["klebsiella", "escherichia"],
             recent_procedures=["urethral_catheter"]),
    _patient("PT-00408", ["sarahmitchell", "jokafor"], "55-64", "Male", "Outpatient clinic",
             recent_antibiotic_classes=["nitrofuran"], days_since_last_antibiotic=60,
             prior_organisms=["proteus"], recent_procedures=["surgical_procedure"]),
    _patient("PT-00410", ["sarahmitchell"], "45-54", "Female", "Outpatient clinic",
             days_since_last_antibiotic=180),
    _patient("PT-00412", ["sarahmitchell", "praman"], "75-84", "Female", "Intensive care unit",
             nursing_home=True, resistant_to=["ciprofloxacin", "cefazolin"], other_resistant_count=1,
             recent_antibiotic_classes=["fluoroquinolone", "glycopeptide"],
             days_since_last_antibiotic=14, prior_organisms=["escherichia", "pseudomonas"],
             recent_procedures=["urethral_catheter", "cvc"]),
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
             prior_organisms=["escherichia"], recent_procedures=["dialysis"]),
]


# Usual adult regimens, used only to make the demo prescriptions look plausible.
REGIMENS = {
    "nitrofurantoin": ("100 mg", "Twice daily", 5),
    "trimethoprim_sulfamethoxazole": ("160/800 mg", "Twice daily", 3),
    "cefazolin": ("1 g IV", "Every 8 hours", 7),
    "ciprofloxacin": ("500 mg", "Twice daily", 7),
    "ceftriaxone": ("1 g IV", "Once daily", 7),
    "piperacillin_tazobactam": ("4.5 g IV", "Every 8 hours", 7),
    "meropenem": ("1 g IV", "Every 8 hours", 7),
}

# The antibiotics on the lab's panel, in the order of the S/R letters in CASES below.
PANEL = ["ampicillin", "nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin",
         "ciprofloxacin", "ceftriaxone", "piperacillin_tazobactam", "meropenem"]

# One prescription and one culture per patient:
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


def load(connection) -> None:
    """Insert the clinicians, the patients, and who looks after whom."""
    clinician_ids = {}
    for person in CLINICIANS:
        result = connection.execute(db.clinicians.insert().values(
            username=person["username"], name=person["name"], role=person["role"],
            password_hash=db.hash_password(person["password"])))
        clinician_ids[person["username"]] = result.inserted_primary_key[0]
    for patient in PATIENTS:
        result = connection.execute(db.patients.insert().values(**patient["row"]))
        patient_id = result.inserted_primary_key[0]
        for username in patient["clinicians"]:
            connection.execute(db.clinician_patients.insert().values(
                clinician_id=clinician_ids[username], patient_id=patient_id))


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
        dose, frequency, duration = REGIMENS[drug]
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
            reviewed_at=datetime.datetime.combine(today, datetime.time(9, 0)) if review else None))
        for antibiotic, letter in zip(PANEL, panel or ""):
            connection.execute(db.susceptibilities.insert().values(
                culture_id=result.inserted_primary_key[0], antibiotic=antibiotic,
                result=_RESULT[letter]))
