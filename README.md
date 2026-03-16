# KNCV Nigeria TB Patient Journey Analysis

An end-to-end data analysis project examining the TB patient journey across KNCV Nigeria's program states — from screening to treatment outcome, with predictive modeling.

---

## Project Overview

This project simulates and analyzes a 60,000-row TB program dataset modeled on KNCV Nigeria's real operations across 20 Nigerian states (2020–2024). It follows the full data science pipeline: data generation, cleaning, exploratory analysis, statistical testing, and machine learning.

**Core Analysis Question:**
> "From screening to success — what drives TB treatment outcomes across KNCV Nigeria's program states?"

---

## Repository Structure

| File | Description |
|------|-------------|
| `generate_kncv_tb_data.py` | Python script to generate the synthetic dataset |
| `kncv_nigeria_tb_data_raw.csv` | Raw generated dataset (60,000 rows, 26 columns) |
| `kncv_nigeria_tb_data_cleaned.csv` | Cleaned dataset after processing |
| `kncv_tb_cleaning_analysis.ipynb` | Data cleaning notebook |
| `kncv_tb_eda.ipynb` | Exploratory data analysis (9 insights) |
| `kncv_tb_statistical_analysis.ipynb` | Statistical tests (Chi-square, Kruskal-Wallis) |
| `kncv_tb_prediction.ipynb` | Logistic regression prediction model |

---

## Dataset

- **Rows:** 60,000 patients
- **Columns:** 26 (+ 1 derived)
- **Date range:** 2020–2024
- **States:** 20 Nigerian states weighted by real TB burden
- **TB Types:** DS-TB, DR-TB, TB/HIV

### Patient Journey Funnel
```
Screened → Presumptive TB → Diagnosed → Confirmed → Treatment → Outcome
```
Patients who are not presumptive, or are ruled out after testing, have null treatment columns by design — this reflects the real program data structure.

### Intentional Data Quality Issues
| Column | Issue |
|--------|-------|
| `age` | Impossible entries (999, 0), missing values |
| `referred_by` | Inconsistent free-text (e.g. "CV", "comm vol", "Community Volunteer") |
| `lost_to_followup_reason` | Inconsistent capitalization and abbreviations |

---

## Key Findings

### EDA
- **Lagos** recorded the highest TB case notifications, consistent with Nigeria's known burden distribution
- **Community Screening** is the leading case finding method
- TB case notifications **dropped significantly in 2020** due to COVID-19 disruption, recovering steadily through 2022–2024
- **DR-TB patients** have the worst treatment outcomes
- **PLHIV** have significantly lower success rates than HIV-negative patients
- **Relocation** is the leading reason for loss to follow-up
- No state met the **WHO 90% treatment success rate target**

### Statistical Analysis
- Significant relationship between TB type and treatment outcome (Chi-square, p < 0.05)
- Significant relationship between PLHIV status and treatment outcome (Chi-square, p < 0.05)
- Significant difference in treatment duration across TB types (Kruskal-Wallis, p < 0.05)

### Prediction Model
- **Model:** Logistic Regression
- **Accuracy:** 59.5% | **ROC-AUC:** 0.58
- **Top predictors:** Treatment regimen, PLHIV status, child under 15
- Class imbalance addressed using `class_weight='balanced'`
- Moderate performance reflects synthetic data limitations — real-world models would benefit from clinical features such as missed doses and weight progression

---

## Tools Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy)
- Power Query
- Power BI
- Git & GitHub

---

## About KNCV Nigeria
KNCV Nigeria is a national NGO dedicated to fighting Tuberculosis across Nigeria, operating in 20 states with funding from USAID, Global Fund, and other partners.

---

*Synthetic dataset generated for portfolio purposes. Modeled on real program structure and Nigerian TB burden data.*