"""The recommendation rule, shared by the app and checked against notebook 04.

The rule: go through the antibiotics from narrowest to broadest spectrum and recommend the first
one whose predicted chance of working is at least the target. If none qualifies, recommend the
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
