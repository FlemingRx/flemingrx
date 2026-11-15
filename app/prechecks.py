"""Checks to make before acting on a recommendation: symptoms, allergies, kidney function, pregnancy.

The model only predicts resistance. These four checks look at what else is on the patient's record
and say whether it gets in the way of the recommended antibiotic, or of treating at all.

THE RULES HERE ARE SIMPLIFIED, FOR THE DEMO. They are common prescribing cautions written as a few
lines of code, not a drug-safety database, and they have not been reviewed by a clinician or a
pharmacist. They flag things for the clinician to look at; they never change the recommendation.
"""

import policy

# The allergy groups a patient's record can list, and how each is shown.
ALLERGY_LABELS = {"penicillin": "Penicillins", "cephalosporin": "Cephalosporins",
                  "carbapenem": "Carbapenems", "sulfonamide": "Sulfonamides",
                  "fluoroquinolone": "Fluoroquinolones", "nitrofurantoin": "Nitrofurantoin"}

# Which allergy group each of the model's antibiotics belongs to.
DRUG_GROUP = {"nitrofurantoin": "nitrofurantoin", "trimethoprim_sulfamethoxazole": "sulfonamide",
              "cefazolin": "cephalosporin", "ciprofloxacin": "fluoroquinolone",
              "ceftriaxone": "cephalosporin", "piperacillin_tazobactam": "penicillin",
              "meropenem": "carbapenem"}
# Other beta-lactams: a penicillin allergy doesn't rule these out, but is worth a second look.
PENICILLIN_RELATED = ("cephalosporin", "carbapenem")

AVOID_IN_PREGNANCY = ("ciprofloxacin", "trimethoprim_sulfamethoxazole")

# How serious a check's finding is, mildest first.
LEVELS = ("ok", "unknown", "caution", "stop")


def run(record: dict, recommended: str) -> list[dict]:
    """The four checks for this patient and this recommended antibiotic (a short drug name).

    Each is {"name", "value", "level", "note"}; level is one of LEVELS.
    """
    return [symptoms(record), allergies(record, recommended), kidney(record, recommended),
            pregnancy(record, recommended)]


def worst(checks: list[dict]) -> str:
    return max((check["level"] for check in checks), key=LEVELS.index)


def _check(name, value, level, note):
    return {"name": name, "value": value, "level": level, "note": note}


def symptoms(record: dict) -> dict:
    found = record.get("symptoms") or []
    if found:
        return _check("Symptoms", ", ".join(found), "ok", "Urinary symptoms are documented.")
    if record.get("pregnancy") == "Pregnant":
        return _check("Symptoms", "None documented", "caution",
                      "No symptoms, but bacteriuria in pregnancy is usually treated.")
    return _check("Symptoms", "None documented", "stop",
                  "A positive culture without symptoms (asymptomatic bacteriuria) usually should "
                  "not be treated. Exceptions include pregnancy and before urological procedures.")


def allergies(record: dict, recommended: str) -> dict:
    recorded = record.get("allergies") or []
    if not recorded:
        return _check("Allergies", "None recorded", "ok", "No antibiotic allergy on record.")
    value = ", ".join(ALLERGY_LABELS.get(name, name.capitalize()) for name in recorded)
    drug, group = policy.drug_label(recommended), DRUG_GROUP.get(recommended)
    if group in recorded:
        return _check("Allergies", value, "stop",
                      f"{drug} conflicts with the recorded {group} allergy.")
    if "penicillin" in recorded and group in PENICILLIN_RELATED:
        return _check("Allergies", value, "caution",
                      f"Penicillin allergy on record: {drug} is a related antibiotic. Check the "
                      f"type of reaction before prescribing.")
    return _check("Allergies", value, "ok", f"Does not affect {drug}.")


def kidney(record: dict, recommended: str) -> dict:
    egfr = record.get("egfr")
    if egfr is None:
        return _check("Kidney function", "No eGFR on record", "unknown",
                      "Check kidney function before choosing a dose.")
    value = f"eGFR {egfr}"
    if egfr >= 60:
        return _check("Kidney function", value, "ok", "No dose adjustment expected.")
    if egfr < 30 and recommended == "nitrofurantoin":
        return _check("Kidney function", value, "stop",
                      "Nitrofurantoin is generally avoided when eGFR is below 30.")
    if egfr < 30:
        return _check("Kidney function", value, "caution",
                      "Severely reduced kidney function: check the dose before prescribing.")
    return _check("Kidney function", value, "caution",
                  "Reduced kidney function: check whether the dose needs adjusting.")


def pregnancy(record: dict, recommended: str) -> dict:
    status = record.get("pregnancy")
    if status is None:
        return _check("Pregnancy", "Not recorded", "unknown", "Confirm pregnancy status if relevant.")
    if status != "Pregnant":
        return _check("Pregnancy", status, "ok", "No pregnancy-related restriction.")
    drug = policy.drug_label(recommended)
    if recommended in AVOID_IN_PREGNANCY:
        return _check("Pregnancy", status, "stop", f"{drug} is generally avoided in pregnancy.")
    if recommended == "nitrofurantoin":
        return _check("Pregnancy", status, "caution", "Nitrofurantoin is generally avoided near term.")
    return _check("Pregnancy", status, "caution", f"Confirm {drug} is suitable at this stage of pregnancy.")
