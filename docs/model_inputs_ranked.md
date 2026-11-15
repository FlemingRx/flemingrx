# Model inputs ranked by importance

How much each of the model's 105 inputs matters, measured on the 2022+ test cultures.

**Method (permutation importance).** For each input, its values are shuffled between patients so it no
longer carries information, and the drop in AUROC is recorded. A bigger drop means the model relies on
that input more. Each figure is the average over 3 shuffles and over 6 antibiotics (meropenem is left
out: about 60 resistant test cultures is too few to rank on). Models: gradient boosting trained on
2008-2021, run locally on the full ARMD data (mean test AUROC 0.738). Notebook 03, section 6b,
produces the same ranking on Databricks and saves it to `workspace.armd.gold_feature_importance`.

**How to read it.**
- Inputs that say nearly the same thing share the credit. "Days since nitrofurantoin", "days since
  nitrofuran subtype" and "nitrofuran class in 90 days" overlap, so each looks smaller alone than the
  information is worth. The group table shuffles a whole group together and is the fairer summary.
- A drop near 0 means removing that one input changes almost nothing, not that the fact is medically
  irrelevant.

## By group

| Group | Inputs | AUROC drop when the whole group is shuffled |
|---|---|---|
| Past resistance | 15 | 0.1043 |
| Antibiotic exposure | 57 | 0.0459 |
| Demographics | 4 | 0.0298 |
| Past organisms | 17 | 0.0286 |
| Where the culture was ordered | 5 | 0.0050 |
| Procedures | 6 | 0.0007 |
| Nursing home | 1 | 0.0005 |

## All 105 inputs

| Rank | Input | Group | Mean AUROC drop | Matters most for |
|---|---|---|---|---|
| 1 | `days_since_resist_ciprofloxacin` | Past resistance | 0.0282 | Ciprofloxacin |
| 2 | `days_since_resist_trimethoprim_sulfamethoxazole` | Past resistance | 0.0259 | TMP-SMX |
| 3 | `days_since_resist_ceftriaxone` | Past resistance | 0.0165 | Ceftriaxone |
| 4 | `prior_org_escherichia` | Past organisms | 0.0146 | Nitrofurantoin |
| 5 | `age_min` | Demographics | 0.0131 | Nitrofurantoin |
| 6 | `days_since_resist_nitrofurantoin` | Past resistance | 0.0121 | Nitrofurantoin |
| 7 | `gender` | Demographics | 0.0109 | Cefazolin |
| 8 | `days_since_resist_cefazolin` | Past resistance | 0.0105 | Cefazolin |
| 9 | `prior_org_pseudomonas` | Past organisms | 0.0090 | Ceftriaxone |
| 10 | `days_since_resist_piperacillin_tazobactam` | Past resistance | 0.0055 | Pip-tazo |
| 11 | `is_inpatient` | Where the culture was ordered | 0.0041 | Ceftriaxone |
| 12 | `days_since_drug_trimethoprim` | Antibiotic exposure | 0.0040 | TMP-SMX |
| 13 | `days_since_subtype_fluoroquinolone` | Antibiotic exposure | 0.0037 | Ciprofloxacin |
| 14 | `days_since_last_antibiotic` | Antibiotic exposure | 0.0033 | Ceftriaxone |
| 15 | `n_prior_resistant_drugs` | Past resistance | 0.0031 | Pip-tazo |
| 16 | `adi_score` | Demographics | 0.0031 | Pip-tazo |
| 17 | `n_antibiotic_days_365d` | Antibiotic exposure | 0.0030 | Nitrofurantoin |
| 18 | `n_prior_cultures_365d` | Past organisms | 0.0022 | Ciprofloxacin |
| 19 | `days_since_drug_nitrofurantoin` | Antibiotic exposure | 0.0020 | TMP-SMX |
| 20 | `days_since_subtype_beta_lactam_combo` | Antibiotic exposure | 0.0020 | Pip-tazo |
| 21 | `days_since_drug_piperacillin_tazobactam` | Antibiotic exposure | 0.0019 | Ciprofloxacin |
| 22 | `days_since_drug_cephalexin` | Antibiotic exposure | 0.0018 | Pip-tazo |
| 23 | `prior_org_klebsiella` | Past organisms | 0.0017 | Nitrofurantoin |
| 24 | `days_since_subtype_cephalosporin_gen3` | Antibiotic exposure | 0.0015 | Ceftriaxone |
| 25 | `days_since_drug_ciprofloxacin` | Antibiotic exposure | 0.0014 | Ciprofloxacin |
| 26 | `abx_class_beta_lactam_90d` | Antibiotic exposure | 0.0014 | Ceftriaxone |
| 27 | `prior_org_proteus` | Past organisms | 0.0008 | Nitrofurantoin |
| 28 | `days_since_subtype_glycopeptide` | Antibiotic exposure | 0.0007 | Nitrofurantoin |
| 29 | `days_since_drug_carbapenem` | Antibiotic exposure | 0.0007 | Ceftriaxone |
| 30 | `days_since_drug_ceftriaxone` | Antibiotic exposure | 0.0006 | Ceftriaxone |
| 31 | `nursing_home` | Nursing home | 0.0006 | Pip-tazo |
| 32 | `days_since_drug_amoxicillin_clavulanate` | Antibiotic exposure | 0.0006 | Ceftriaxone |
| 33 | `days_since_drug_cefazolin` | Antibiotic exposure | 0.0006 | TMP-SMX |
| 34 | `ward_er` | Where the culture was ordered | 0.0006 | Ceftriaxone |
| 35 | `days_since_subtype_sulfonamide_combo` | Antibiotic exposure | 0.0005 | Pip-tazo |
| 36 | `adi_state_rank` | Demographics | 0.0005 | Pip-tazo |
| 37 | `days_since_drug_levofloxacin` | Antibiotic exposure | 0.0004 | Pip-tazo |
| 38 | `days_since_subtype_penicillin` | Antibiotic exposure | 0.0004 | Pip-tazo |
| 39 | `days_since_subtype_nitroimidazole` | Antibiotic exposure | 0.0004 | Cefazolin |
| 40 | `prior_org_citrobacter` | Past organisms | 0.0004 | TMP-SMX |
| 41 | `days_since_subtype_lincosamide` | Antibiotic exposure | 0.0003 | TMP-SMX |
| 42 | `days_since_subtype_macrolide` | Antibiotic exposure | 0.0003 | Pip-tazo |
| 43 | `days_since_drug_amoxicillin_ampicillin` | Antibiotic exposure | 0.0003 | Pip-tazo |
| 44 | `prior_org_enterococcus` | Past organisms | 0.0003 | Ceftriaxone |
| 45 | `days_since_subtype_tetracycline` | Antibiotic exposure | 0.0003 | TMP-SMX |
| 46 | `proc_urethral_catheter` | Procedures | 0.0002 | Nitrofurantoin |
| 47 | `days_since_subtype_aminoglycoside` | Antibiotic exposure | 0.0002 | Ceftriaxone |
| 48 | `abx_class_ansamycin_90d` | Antibiotic exposure | 0.0002 | - |
| 49 | `proc_cvc` | Procedures | 0.0002 | - |
| 50 | `proc_surgical_procedure` | Procedures | 0.0002 | - |
| 51 | `days_since_subtype_cephalosporin_gen1` | Antibiotic exposure | 0.0001 | - |
| 52 | `prior_org_enterobacter` | Past organisms | 0.0001 | - |
| 53 | `proc_mechvent` | Procedures | 0.0001 | - |
| 54 | `days_since_subtype_urinary_antiseptic` | Antibiotic exposure | 0.0001 | - |
| 55 | `days_since_drug_oral_cephalosporin_gen3` | Antibiotic exposure | 0.0001 | - |
| 56 | `prior_resist_nitrofurantoin` | Past resistance | 0.0001 | - |
| 57 | `days_since_subtype_cephalosporin_gen2` | Antibiotic exposure | 0.0001 | - |
| 58 | `days_since_subtype_ansamycin` | Antibiotic exposure | 0.0001 | - |
| 59 | `days_since_subtype_fosfomycin` | Antibiotic exposure | 0.0000 | - |
| 60 | `prior_resist_ceftriaxone` | Past resistance | 0.0000 | - |
| 61 | `prior_org_morganella` | Past organisms | 0.0000 | - |
| 62 | `ward_op` | Where the culture was ordered | 0.0000 | - |
| 63 | `abx_class_macrolide_lincosamide_90d` | Antibiotic exposure | 0.0000 | - |
| 64 | `prior_resist_piperacillin_tazobactam` | Past resistance | 0.0000 | - |
| 65 | `proc_parenteral_nutrition` | Procedures | 0.0000 | - |
| 66 | `prior_resist_trimethoprim_sulfamethoxazole` | Past resistance | 0.0000 | - |
| 67 | `prior_org_cons` | Past organisms | 0.0000 | - |
| 68 | `days_since_drug_cefepime` | Antibiotic exposure | 0.0000 | - |
| 69 | `abx_class_aminoglycoside_90d` | Antibiotic exposure | 0.0000 | - |
| 70 | `abx_class_nitroimidazole_90d` | Antibiotic exposure | 0.0000 | - |
| 71 | `days_since_subtype_oxazolidinone` | Antibiotic exposure | 0.0000 | - |
| 72 | `abx_class_tetracycline_90d` | Antibiotic exposure | 0.0000 | - |
| 73 | `abx_class_glycopeptide_90d` | Antibiotic exposure | 0.0000 | - |
| 74 | `prior_resist_ciprofloxacin` | Past resistance | 0.0000 | - |
| 75 | `prior_org_serratia` | Past organisms | 0.0000 | - |
| 76 | `abx_class_combination_antibiotic_90d` | Antibiotic exposure | 0.0000 | - |
| 77 | `prior_org_streptococcus` | Past organisms | 0.0000 | - |
| 78 | `ward_ip` | Where the culture was ordered | 0.0000 | - |
| 79 | `abx_class_oxazolidinone_90d` | Antibiotic exposure | 0.0000 | - |
| 80 | `abx_class_antitubercular_90d` | Antibiotic exposure | 0.0000 | - |
| 81 | `abx_class_sulfonamide_90d` | Antibiotic exposure | 0.0000 | - |
| 82 | `abx_class_monobactam_90d` | Antibiotic exposure | 0.0000 | - |
| 83 | `abx_class_folate_synthesis_inhibitor_90d` | Antibiotic exposure | 0.0000 | - |
| 84 | `prior_org_stenotrophomonas` | Past organisms | 0.0000 | - |
| 85 | `prior_org_acinetobacter` | Past organisms | 0.0000 | - |
| 86 | `prior_org_providencia` | Past organisms | 0.0000 | - |
| 87 | `prior_resist_meropenem` | Past resistance | 0.0000 | - |
| 88 | `days_since_subtype_antitubercular` | Antibiotic exposure | 0.0000 | - |
| 89 | `days_since_subtype_carbapenem` | Antibiotic exposure | 0.0000 | - |
| 90 | `days_since_subtype_cephalosporin_gen4` | Antibiotic exposure | 0.0000 | - |
| 91 | `days_since_subtype_monobactam` | Antibiotic exposure | 0.0000 | - |
| 92 | `days_since_subtype_nitrofuran` | Antibiotic exposure | 0.0000 | - |
| 93 | `days_since_subtype_polymyxin` | Antibiotic exposure | 0.0000 | - |
| 94 | `days_since_resist_meropenem` | Past resistance | 0.0000 | - |
| 95 | `proc_dialysis` | Procedures | -0.0000 | - |
| 96 | `abx_class_fosfomycin_90d` | Antibiotic exposure | -0.0000 | - |
| 97 | `abx_class_fluoroquinolone_90d` | Antibiotic exposure | -0.0000 | - |
| 98 | `prior_org_candida` | Past organisms | -0.0000 | - |
| 99 | `days_since_subtype_sulfonamide` | Antibiotic exposure | -0.0000 | - |
| 100 | `abx_class_nitrofuran_90d` | Antibiotic exposure | -0.0000 | - |
| 101 | `prior_org_staphylococcus` | Past organisms | -0.0000 | - |
| 102 | `abx_class_urinary_antiseptic_90d` | Antibiotic exposure | -0.0000 | - |
| 103 | `prior_resist_cefazolin` | Past resistance | -0.0000 | - |
| 104 | `days_since_subtype_folate_synthesis_inhibitor` | Antibiotic exposure | -0.0000 | - |
| 105 | `ward_icu` | Where the culture was ordered | -0.0001 | - |
