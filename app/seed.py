"""Fabricated demo data: three clinicians and the patients assigned to them.

NONE OF THIS IS REAL. The clinicians and patients are made up for the hackathon demo, and the
passwords below are demo passwords, here so anyone running the demo can sign in. Do not reuse
them anywhere, and replace this file before the app goes near real data.

db.init() loads this once, into an empty database. To reload it after editing, empty the three
tables (in the Lakebase SQL editor: TRUNCATE flemingrx.clinicians, flemingrx.patients CASCADE)
and start the app again.
"""

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
