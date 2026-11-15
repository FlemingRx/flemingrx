"""The recommendation rule, shared by the app and checked against notebook 04.

The rule: go through the antibiotics from narrowest to broadest spectrum and recommend the first
one whose predicted chance of working is at least the target. The target (the "bar") depends on
how sick the patient is: see the severity block below. If none qualifies, recommend the
broadest. tests/test_policy.py checks this gives the same answer as notebook 04's `recommend`.
"""

# Display names, narrowest -> broadest spectrum (the order notebook 02 defines).
DRUG_LABELS = {
    "nitrofurantoin": "Nitrofurantoin",
    "trimethoprim_sulfamethoxazole": "Trimethoprim/Sulfamethoxazole",
    "cefazolin": "Cefazolin",
    "ciprofloxacin": "Ciprofloxacin",
    "ceftriaxone": "Ceftriaxone",
    "piperacillin_tazobactam": "Piperacillin/Tazobactam",
    "meropenem": "Meropenem",
}

# Spectrum groups, the same as notebook 04 reports on.
NARROW_DRUGS = ["nitrofurantoin", "trimethoprim_sulfamethoxazole", "cefazolin"]
BROAD_DRUGS = ["piperacillin_tazobactam", "meropenem"]


# A usual adult regimen for each antibiotic: (dose, frequency, days). It is what gets recorded when
# a clinician accepts or overrides a recommendation, as a starting point they can then adjust.
# Simplified for the demo and not reviewed by a pharmacist: no adjustment for kidney function,
# weight or the site and severity of infection.
USUAL_REGIMENS = {
    "nitrofurantoin": ("100 mg", "Twice daily", 5),
    "trimethoprim_sulfamethoxazole": ("160/800 mg", "Twice daily", 3),
    "cefazolin": ("1 g IV", "Every 8 hours", 7),
    "ciprofloxacin": ("500 mg", "Twice daily", 7),
    "ceftriaxone": ("1 g IV", "Once daily", 7),
    "piperacillin_tazobactam": ("4.5 g IV", "Every 8 hours", 7),
    "meropenem": ("1 g IV", "Every 8 hours", 7),
}


# --- Severity (notebook 04, section 5b) ---
# How sick the patient looks decides how much risk of failure is acceptable, so each severity level
# has its own bar. The level is worked out here; the bars themselves are read from the model's
# MLflow run (data.severity_bars). DEFAULT_SEVERITY_BARS is only used if that can't be read.
# Everything in this block mirrors notebook 04, and tests/test_policy.py checks it still does.
SEVERITY_LEVELS = ["Mild", "Systemic signs", "Severe"]
DEFAULT_SEVERITY_BARS = {"Mild": 0.85, "Systemic signs": 0.90, "Severe": 0.95}

# Vital-sign cut-offs (standard sepsis-screening values; temperature is in Fahrenheit)
FEVER_F = 100.4             # temperature at or above this = fever
FAST_HEART = 90             # heart rate above this
FAST_BREATHING = 20         # breaths per minute above this
VERY_FAST_BREATHING = 22    # with LOW_NORMAL_SBP: two warning signs at once -> severe
LOW_NORMAL_SBP = 100        # systolic blood pressure at or below this
LOW_SBP = 90                # systolic blood pressure below this = low blood pressure


def severity_level(icu, ward_ip, ward_er, temperature_f=None, heart_rate=None, respiratory_rate=None,
                   systolic_bp=None) -> int:
    """Position in SEVERITY_LEVELS for one patient (0 = mild, 1 = systemic signs, 2 = severe).

    Care-setting flags are 1 for yes. A vital sign that wasn't recorded (None) counts as not abnormal.
      severe:   intensive care, or low blood pressure, or very fast breathing with low-normal pressure
      systemic: fever, or two of (fever, fast heart rate, fast breathing), or admitted to a ward
      mild:     everyone else
    """
    fever = temperature_f is not None and temperature_f >= FEVER_F
    fast_heart = heart_rate is not None and heart_rate > FAST_HEART
    fast_breathing = respiratory_rate is not None and respiratory_rate > FAST_BREATHING
    low_pressure = systolic_bp is not None and systolic_bp < LOW_SBP
    breathing_and_pressure = (respiratory_rate is not None and systolic_bp is not None
                              and respiratory_rate >= VERY_FAST_BREATHING
                              and systolic_bp <= LOW_NORMAL_SBP)
    if icu == 1 or low_pressure or breathing_and_pressure:
        return 2
    if fever or (fever + fast_heart + fast_breathing) >= 2 or (ward_ip == 1 and ward_er != 1):
        return 1
    return 0


def severity_bar_param(level: str) -> str:
    """The name notebook 04 logs a level's bar under in MLflow, e.g. severity_bar_systemic_signs."""
    return "severity_bar_" + level.lower().replace(" ", "_")


def drug_label(slug: str) -> str:
    return DRUG_LABELS.get(slug, slug.replace("_", " ").title())


def spectrum(slug: str) -> str:
    if slug in NARROW_DRUGS:
        return "Narrow"
    if slug in BROAD_DRUGS:
        return "Broad"
    return "Medium"


def recommend(p_resistant, target_success: float) -> int:
    """Position of the narrowest drug that is likely enough to work (the broadest if none is).

    `p_resistant` is the predicted chance of resistance for each drug, narrowest first.
    """
    max_risk = 1 - target_success
    for position, risk in enumerate(p_resistant):
        if risk <= max_risk:
            return position
    return len(p_resistant) - 1


def explain(p_resistant, drugs, target_success: float) -> str:
    """One sentence on why the rule picked the drug it did."""
    picked = recommend(p_resistant, target_success)
    chance = 1 - p_resistant[picked]
    name = drug_label(drugs[picked])
    if p_resistant[picked] > 1 - target_success:
        return (f"No antibiotic reaches a {target_success:.0%} chance of working, so the broadest "
                f"one is recommended: {name} ({chance:.0%}).")
    if picked == 0:
        return (f"{name} is the narrowest option and already has a {chance:.0%} chance of working, "
                f"which clears the {target_success:.0%} bar.")
    skipped = ", ".join(drug_label(d) for d in drugs[:picked])
    return (f"{name} is the narrowest antibiotic with at least a {target_success:.0%} chance of "
            f"working ({chance:.0%}). Narrower options fall short: {skipped}.")
