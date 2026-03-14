"""
KNCV Nigeria TB Program Dataset Generator
==========================================
Generates 60,000 rows of realistic TB program data
following the actual patient journey:
Screening → Diagnosis → Treatment → Outcome

Dirty columns (realistic free-text inconsistencies):
- age (self-reported: missing values, impossible entries, teenagers rounding up)
- referred_by (free-text: inconsistent spelling/abbreviations)
- lost_to_followup_reason (free-text: inconsistent capitalization, vague entries)

Note: treatment_duration_months is NOT stored — it will be derived
during analysis as: (outcome_date - treatment_start_date) / 30
This is more realistic since duration is never manually entered in real programs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

N = 60000

# ── STATE + LGA CONFIGURATION ─────────────────────────────────────────────────

STATE_WEIGHTS = {
    "Lagos":       0.16,
    "Kano":        0.12,
    "Oyo":         0.09,
    "Kaduna":      0.07,
    "Ogun":        0.06,
    "Osun":        0.05,
    "Katsina":     0.05,
    "Bauchi":      0.04,
    "Nasarawa":    0.04,
    "Benue":       0.04,
    "Taraba":      0.03,
    "Anambra":     0.03,
    "Imo":         0.03,
    "Delta":       0.03,
    "Akwa Ibom":   0.03,
    "Cross River": 0.03,
    "Rivers":      0.03,
    "Plateau":     0.03,
    "Enugu":       0.02,
    "Kwara":       0.02,
}

STATE_LGAS = {
    "Lagos": [
        "Alimosho", "Ikeja", "Surulere", "Mushin", "Kosofe",
        "Oshodi-Isolo", "Agege", "Ajeromi-Ifelodun", "Lagos Island",
        "Shomolu", "Somolu", "Mainland", "Apapa", "Eti-Osa", "Ikorodu"
    ],
    "Kano": [
        "Kano Municipal", "Fagge", "Dala", "Gwale", "Nassarawa",
        "Tarauni", "Ungogo", "Kumbotso", "Dawakin Kudu", "Warawa"
    ],
    "Oyo": [
        "Ibadan North", "Ibadan South-West", "Ibarapa East",
        "Ogbomoso North", "Akinyele", "Ibadan North-East",
        "Ibadan South-East", "Egbeda", "Ona Ara", "Lagelu"
    ],
    "Kaduna": [
        "Chikun", "Kaduna North", "Kaduna South", "Igabi",
        "Zaria", "Sabon Gari", "Giwa", "Ikara", "Kajuru"
    ],
    "Ogun": [
        "Abeokuta South", "Abeokuta North", "Ifo", "Sagamu",
        "Ado-Odo/Ota", "Obafemi-Owode", "Ewekoro", "Ijebu Ode"
    ],
}

def get_lga(state):
    if state in STATE_LGAS:
        return random.choice(STATE_LGAS[state])
    return f"{state} Central LGA"


# ── DATE HELPERS ──────────────────────────────────────────────────────────────

def random_date(start_year=2020, end_year=2024):
    """
    Weighted random date 2020-2024.
    2020-2021: fewer cases (COVID disruption to TB services).
    2022-2024: recovery and scale-up.
    Reflects real-world impact of COVID-19 on TB programs in Nigeria.
    """
    year_weights = {
        2020: 0.10,  # COVID hit hard, TB services severely disrupted
        2021: 0.15,  # partial recovery
        2022: 0.22,  # services resuming
        2023: 0.27,  # scale-up
        2024: 0.26,  # sustained high activity
    }
    year = np.random.choice(
        list(year_weights.keys()),
        p=list(year_weights.values())
    )
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def add_days(date, days):
    return date + timedelta(days=days)


# ── DIRTY DATA HELPERS ────────────────────────────────────────────────────────

REFERRED_BY_VARIANTS = {
    "Community Volunteer": [
        "Community Volunteer", "community volunteer", "Comm. Vol.",
        "comm vol", "Community Vol", "CV", "community vol.",
        "Comunity Volunteer", "COMMUNITY VOLUNTEER"
    ],
    "Health Worker": [
        "Health Worker", "health worker", "HW", "H/Worker",
        "Hlth Worker", "health wrkr", "HEALTH WORKER", "Htlh Worker"
    ],
    "Self Referral": [
        "Self Referral", "self referral", "Self", "self",
        "Walk-in", "walk in", "SELF REFERRAL", "Self-referral"
    ],
    "Contact Tracing": [
        "Contact Tracing", "contact tracing", "CT", "Contact trace",
        "Cont. Tracing", "contact tracer", "CONTACT TRACING"
    ],
    "PLHIV Clinic": [
        "PLHIV Clinic", "plhiv clinic", "HIV Clinic", "ART Clinic",
        "PLHIV clinic", "art clinic", "ART/PLHIV", "plhiv"
    ],
}

def dirty_referred_by():
    category = random.choice(list(REFERRED_BY_VARIANTS.keys()))
    return random.choice(REFERRED_BY_VARIANTS[category])

LTFU_REASON_VARIANTS = [
    "Relocated", "relocated", "RELOCATED", "Moved away", "moved",
    "Moved to another state", "relocated to village",
    "Refused treatment", "refused", "Refused", "REFUSED TREATMENT",
    "patient refused", "Refused to continue",
    "Unknown", "unknown", "UNKNOWN", "Don't know", "not known", "NK",
    "Financial constraints", "financial", "No money", "cant afford",
    "financial issues", "FINANCIAL CONSTRAINTS",
    "Feeling better", "felt better", "Feeling Better", "feels fine",
    "said he felt better", "patient felt better",
    "Side effects", "side effects", "SIDE EFFECTS", "drug side effects",
    "adverse reaction", "Side Effects",
]

def dirty_ltfu_reason():
    return random.choice(LTFU_REASON_VARIANTS)

def dirty_age(real_age):
    roll = random.random()
    if roll < 0.005:
        return 999       # clearly wrong entry
    elif roll < 0.01:
        return 0         # impossible entry
    elif roll < 0.02:
        return None      # missing
    elif roll < 0.04 and real_age < 20:
        return 18        # teenagers rounding up
    return real_age


# ── MAIN GENERATION ───────────────────────────────────────────────────────────

def generate_dataset(n=60000):
    records = []
    states = list(STATE_WEIGHTS.keys())
    state_probs = list(STATE_WEIGHTS.values())

    age_probs = np.concatenate([
        np.full(14, 0.008),  # ages 1-14
        np.full(50, 0.017),  # ages 15-64
        np.full(16, 0.006),  # ages 65-80
    ])
    age_probs = age_probs / age_probs.sum()

    for i in range(n):
        record = {}

        # PATIENT ID
        record["patient_id"] = f"KNCV-{str(i+1).zfill(6)}"

        # DEMOGRAPHICS
        state = np.random.choice(states, p=state_probs)
        record["state"] = state
        record["LGA"] = get_lga(state)
        record["sex"] = np.random.choice(["Male", "Female"], p=[0.58, 0.42])
        real_age = int(np.random.choice(range(1, 81), p=age_probs))
        record["age"] = dirty_age(real_age)

        # SCREENING
        screening_date = random_date(2020, 2024)
        record["screening_date"] = screening_date.strftime("%Y-%m-%d")
        record["case_finding_method"] = np.random.choice(
            ["Community Screening", "Index Testing", "Facility-Based", "Digital X-ray"],
            p=[0.35, 0.25, 0.28, 0.12]
        )
        record["referred_by"] = dirty_referred_by()

        # PRESUMPTIVE TB
        is_presumptive = np.random.choice([True, False], p=[0.80, 0.20])
        record["presumptive_tb"] = "Yes" if is_presumptive else "No"

        # KEY POPULATIONS
        is_plhiv = np.random.choice([True, False], p=[0.22, 0.78])
        is_child = real_age < 15
        is_contact = np.random.choice([True, False], p=[0.30, 0.70])
        is_high_risk = np.random.choice([True, False], p=[0.15, 0.85])

        record["is_plhiv"] = "Yes" if is_plhiv else "No"
        record["is_child_under15"] = "Yes" if is_child else "No"
        record["is_contact_of_tb_patient"] = "Yes" if is_contact else "No"
        record["is_high_risk_setting"] = "Yes" if is_high_risk else "No"

        # FUNNEL: NOT PRESUMPTIVE — skip diagnosis + treatment
        if not is_presumptive:
            for col in ["diagnostic_tool", "diagnosis_date", "tb_type", "specimen_type",
                        "treatment_start_date", "treatment_regimen", "dots_method",
                        "treatment_facility_level", "treatment_outcome",
                        "outcome_date", "lost_to_followup_reason"]:
                record[col] = None

            tpt_eligible = is_plhiv or is_contact
            record["tpt_eligible"] = "Yes" if tpt_eligible else "No"
            record["tpt_initiated"] = (
                np.random.choice(["Yes", "No"], p=[0.62, 0.38]) if tpt_eligible else "No"
            )
            records.append(record)
            continue

        # DIAGNOSIS
        diagnosis_date = add_days(screening_date, random.randint(1, 14))
        record["diagnosis_date"] = diagnosis_date.strftime("%Y-%m-%d")
        record["diagnostic_tool"] = np.random.choice(
            ["GeneXpert", "Truenat", "TB LAMP", "Smear Microscopy"],
            p=[0.45, 0.20, 0.15, 0.20]
        )
        record["specimen_type"] = np.random.choice(
            ["Sputum", "Stool", "Blood", "Urine"],
            p=[0.70, 0.10, 0.12, 0.08]
        )

        # FUNNEL: RULED OUT — no treatment
        is_confirmed = np.random.choice([True, False], p=[0.75, 0.25])
        if not is_confirmed:
            for col in ["tb_type", "treatment_start_date", "treatment_regimen",
                        "dots_method", "treatment_facility_level", "treatment_outcome",
                        "outcome_date", "lost_to_followup_reason"]:
                record[col] = None

            tpt_eligible = is_plhiv or is_contact
            record["tpt_eligible"] = "Yes" if tpt_eligible else "No"
            record["tpt_initiated"] = (
                np.random.choice(["Yes", "No"], p=[0.62, 0.38]) if tpt_eligible else "No"
            )
            records.append(record)
            continue

        # CONFIRMED TB PATIENT
        if is_plhiv:
            tb_type = np.random.choice(["DS-TB", "DR-TB", "TB/HIV"], p=[0.25, 0.10, 0.65])
        elif is_child:
            tb_type = np.random.choice(["DS-TB", "DR-TB", "TB/HIV"], p=[0.80, 0.10, 0.10])
        else:
            tb_type = np.random.choice(["DS-TB", "DR-TB", "TB/HIV"], p=[0.65, 0.15, 0.20])
        record["tb_type"] = tb_type

        treatment_start = add_days(diagnosis_date, random.randint(2, 7))
        record["treatment_start_date"] = treatment_start.strftime("%Y-%m-%d")

        if tb_type == "DS-TB":
            record["treatment_regimen"] = "2HRZE/4HR"
            expected_duration_days = 6 * 30
        elif tb_type == "DR-TB":
            record["treatment_regimen"] = np.random.choice(
                ["BPaLM", "Shorter DR-TB Regimen", "Longer DR-TB Regimen"],
                p=[0.40, 0.35, 0.25]
            )
            expected_duration_days = random.randint(18, 24) * 30
        else:  # TB/HIV
            record["treatment_regimen"] = "2HRZE/4HR + ART"
            expected_duration_days = 6 * 30

        record["dots_method"] = np.random.choice(
            ["99DOTS", "Video Observed Therapy", "Facility DOTS", "Community DOTS"],
            p=[0.30, 0.20, 0.35, 0.15]
        )
        record["treatment_facility_level"] = np.random.choice(
            ["Primary", "Secondary", "Tertiary"],
            p=[0.45, 0.40, 0.15]
        )

        # TPT not applicable for confirmed TB
        record["tpt_eligible"] = None
        record["tpt_initiated"] = None

        # OUTCOME — weighted by tb_type and population
        if tb_type == "DR-TB":
            outcome_probs = [0.55, 0.15, 0.20, 0.05, 0.05]
        elif is_plhiv or tb_type == "TB/HIV":
            outcome_probs = [0.62, 0.12, 0.15, 0.06, 0.05]
        elif is_child:
            outcome_probs = [0.70, 0.12, 0.10, 0.04, 0.04]
        else:
            outcome_probs = [0.75, 0.08, 0.10, 0.04, 0.03]

        outcome_probs = np.array(outcome_probs)
        outcome_probs = outcome_probs / outcome_probs.sum()

        outcome = np.random.choice(
            ["Treatment Success", "Died", "Lost to Follow-up", "Treatment Failed", "Not Evaluated"],
            p=outcome_probs
        )
        record["treatment_outcome"] = outcome

        # outcome_date calculated from treatment start + expected duration
        outcome_date = add_days(treatment_start, expected_duration_days + random.randint(-10, 10))
        record["outcome_date"] = outcome_date.strftime("%Y-%m-%d")

        record["lost_to_followup_reason"] = (
            dirty_ltfu_reason() if outcome == "Lost to Follow-up" else None
        )

        records.append(record)

    return pd.DataFrame(records)


# ── DUPLICATE RECORDS ─────────────────────────────────────────────────────────

def add_duplicates(df, n_dupes=150):
    """Duplicate ~150 records simulating double data entry"""
    dupe_indices = np.random.choice(df.index, size=n_dupes, replace=False)
    dupes = df.loc[dupe_indices].copy()
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ── RUN ───────────────────────────────────────────────────────────────────────

print("Generating 60,000 TB patient records...")
df = generate_dataset(N)

print("Adding duplicate records...")
df = add_duplicates(df, n_dupes=150)

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nTreatment outcomes:")
print(df["treatment_outcome"].value_counts(dropna=False))
print(f"\nTB types:")
print(df["tb_type"].value_counts(dropna=False))
print(f"\nTop 10 states:")
print(df["state"].value_counts().head(10))
print(f"\nNull counts per column:")
print(df.isnull().sum())

output_path = "/mnt/user-data/outputs/kncv_nigeria_tb_data_raw.csv"
df.to_csv(output_path, index=False)
print(f"\nDataset saved to {output_path}")
