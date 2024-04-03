# Healthcare Analytics Competition: Medication Non-Adherence in Oncology

## Table of Contents

- [Background Information](#background-information)
- [Main Objectives](#main-objectives)
- [Data Sources](#data-sources)
- [Initial Hypothesis](#initial-hypothesis)
- [Usage](#usage)
- [Confidentiality](#confidentiality)

---

## Background Information

The medication under study has proven effectiveness but also causes various side effects, leading to low adherence
rates. [See complete list of side effects](#side-effects-of-osimertinib).

---

## Main Objectives

- **Motivation and Opportunity**: Understand the factors contributing to low medication adherence rates in oncology.

- **Predictive Modeling Target**: Build a predictive model to identify patients likely to discontinue the medication due
  to Adverse Drug Events (ADEs).

---

## Data Sources

- **Target Data**: Includes therapy start and end dates, as well as demographic attributes.

- **Medical Claims Data**: Contains information recorded before and during the therapy, including diagnostic codes.

- **Pharmacy Claims Data**: Includes pharmacy claims information and indicators for drugs that interact with the
  medication under study.

---

## Initial Hypothesis

The data must meet two essential criteria to be considered usable:

1. Reported an ADE.
2. Ends before 180 days.

Failure to meet any of these criteria renders the data unusable for this study.

---

## Usage

**Step 1**: Clone the repository to your local machine.

**Step 2**: Place all your datasets in the `data/` directory.

**Step 3**: Run the Python script named `mask_data.py` to mask the sensitive information.
