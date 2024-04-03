# 2023 Humana Mays Healthcare Analytics Case Competition Problem Prompt

**Author:** *Dustin James Harper—Senior Data Scientist, Humana Pharmacy Analytics and Consulting*

This document outlines the problem statement and available data for the **2023 Humana Mays Healthcare Analytics Case
Competition**. For information about registration, schedule, submission logistics, and the leaderboard,
see [Humana TAMU Analytics](https://mays.tamu.edu/humana-tamu-analytics/).

# We utilized Jupyter Notebook for the project. 

Our primary Jupyter Notebook for this project is located [here](https://github.com/osu-msba/ban5753-fall2023-rxdata/blob/main/Humana-Mays%20Competition/Humana_Confidential_Analytics/Model%20Building/LightGBM/Final_Model_deliverable.ipynb).



## 1. MOTIVATION AND OPPORTUNITY

Oncology is a clinical area of focus that has seen significant advances in research and new therapies. Despite these
advances, cancer remains a leading cause of death, with approximately 600,000 people dying from cancer each year in the
US alone. While new treatments are coming to the market, many of them are associated with potentially significant side
effects which are a barrier to people staying adherent to their life-saving medications.

One of these medications is Osimertinib, an oral tyrosine kinase inhibitor used for patients in early-stage lung cancer
of the non-small cell variety, with a specific targetable mutation known as EGFR. Osimertinib is known to be a largely
effective medication, with patients receiving it being twice as likely to survive compared to those who do not take the
medication. Additionally, patients taking the medication as prescribed are 80% less likely to have a recurrence of their
cancer.

As with many other oncology drugs, tolerance of Osimertinib can be difficult due to the side effects associated with the
medication—specifically nausea, fatigue, pain, high blood glucose, and constipation. Many of these side effects are
manageable with proper counseling and avoidance techniques, but many patients may opt to discontinue their treatment
rather than seeking guidance on how to manage them. Approximately one quarter of Humana members taking Osimertinib have
side effects and discontinue their Osimertinib therapy within the first 6 months.

To address this problem, we need to leverage our data and analytics to target members at risk, encourage medication
adherence, and allow our oncology patients to live longer, fulfilling lives.

## 2. PREDICTIVE MODELING TARGET

As explained previously, taking Osimertinib as prescribed can help people live longer, but treatable side-effects might
cause people to end their treatment prematurely. To this end, your assignment is to build a model to predict which
therapies will end prematurely after a reported side effect, also known as an adverse drug event or ADE.

It is in everyone's best interest for a patient to keep taking their Osimertinib therapy for as long as possible. In
this case, a successful therapy is six months (180 days) of continuous Osimertinib therapy. Conversely, an unsuccessful
therapy is any therapy that ends before 180 days. However, to specifically target members who may be discontinuing due
to experiencing an ADE, we've defined the target more specifically to include an ADE at some point during the
unsuccessful therapy. In the training data, this target is recorded in the column labeled `tgt_ade_dc_ind`.

### 2.1 UNSUCCESSFUL THERAPY: TGT_ADE_DC_IND==1

The target is defined as a therapy that ends prematurely (before 180 days) and has an ADE reported at some time during
the therapy. The target definition has already been done for you and is available in `target_train.csv` as the column '
tgt_ade_dc_ind'.

### 2.2 ALL OTHER THERAPIES: TGT_ADE_DC_IND==0

Since the target definition is so specific, there are several other types of therapies not included in the target group.
The following are a few examples:

- Successful therapies with 180 days of continuous treatment
- Therapies that end prematurely with no reported ADEs
- Therapies where the member changes to another insurance plan or dies before 180 days

## 3. AVAILABLE DATA

Since we're trying to predict if a therapy will end prematurely after an ADE, our data is organized based on a specific
therapy with one member, a start date and end date.

Each category of data is separated into a train and holdout set. Use the train set to train your model and submit your
predictions on the holdout set for scoring.

### 3.1 FILE DESCRIPTIONS

The following sections provide a brief overview of the available data. Detailed descriptions of all fields are available
in `data_dictionary.csv`. With the exception of the `target_holdout`, the target and holdout sets contain the exact same
data columns, but for different sets of individuals.

#### 3.1.1 Target: `target_train` (1232 records), `target_holdout` (420 records)

Unique on the person identifier and therapy identifier. Contains information about the therapy start and end dates, the
target identifier, and protected attributes for the individual (sex, race, age, etc.)

Important Note: When you submit your results for Round 1, you will need to submit an ID, score and rank for each
individual ID in the `target_holdout` file. The ID will come directly from `target_holdout.csv`, and the score and rank
will come from your predictive model. You will notice that there is no target identifier or therapy end date included
in `target_holdout.csv`.

Sum of `tgt_ade_dc_ind` in `target_train.csv`: 117

#### 3.1.2 Medical Claims: `medclms_train` (100159 records), `medclms_holdout` (23232 records)

Unique on claim identifier. Contains simplified information about all medical claims for an individual during the time
90 days before their Osimertinib therapy and through the end of therapy. This data includes visit and process dates,
diagnosis codes and indicators for diagnosis codes of interest. E.g. since nausea is a known side-effect of Osimertinib,
we added an indicator column for a diagnosis code related to nausea.

Sums from `medclms_train`:

- `ade_diagnosis`: 6848
- `seizure_diagnosis`: 333
- `pain_diagnosis`: 63

Sums from `medclms_holdout`:

- `ade_diagnosis`: 1841
- `seizure_diagnosis`: 37
- `pain_diagnosis`: 24

#### 3.1.3 Pharmacy Claims: `rxclms_train` (32133 records), `rxclms_holdout` (6670 records)

Unique on claim identifier. Contains simplified information about all pharmacy claims for an individual during the time
90 days before their Osimertinib therapy and through the end of therapy. This data includes service and process dates,
drug identifier codes (NDC) and indicators for drug codes of interest. E.g. since anticoagulants are known to have drug
interactions with Osimertinib, we added an indicator column for a drug code for an anticoagulant.

Average `rx_cost` in `rxclms_train`: 2463.950
Average `rx_cost` in `rxclms_holdout`: 2159.679

#### 3.1.4 Data Dictionary: `data_dictionary.csv` (49 records)

Contains a description for each data column available in the claims datasets.

#### 3.1.5 Race Code Descriptions: `race_cd_desc.csv` (7 records)

Contains definitions for the coded race codes in the target files.

# Project Tasks and Team Responsibilities

This README outlines the key tasks and team responsibilities for our upcoming project. The agenda will be discussed in detail in a meeting with Nathan. Below, you'll find a breakdown of the work we have planned.

## Upcoming Meetings

- Schedule an initial meeting with Nathan to discuss the agenda.
- Plan a follow-up meeting later this week to delve into the data once access is granted.

## Research Responsibilities

### Initial Research Allocation

- **Bella**: Focus on FCA (Factor Component Analysis).
- **Nathan**: Examine variables featuring a high number of categories.
- **John**: Study the Disparity Index, understanding its implications when the value is 1.00 (identical risk between two groups), less than 1.00 (lower risk in selected group), or greater than 1.00 (higher risk in selected group).
- **Brock**: Investigate approaches for managing missing data values.

## Development Tasks

- Create a README file detailing programming instructions.
- Produce a TWBX file for visual data representations in Tableau.

## Security and Confidentiality

- Consider utilizing variable masking (X1, X2, ..., Xn) for secure communication.
- Python offers anonymization features; explore them.
- Ensure that the GitHub repository remains private for project confidentiality.

## Time Management

- Balance time commitments between two ongoing competitions.
- Also balance between the initial submission and final report.

## Team Skills and Responsibilities

### Brock

- **Specialization**: Programming and data visualization.
- **Expertise**: Python, Tableau, Jupyter Notebook, GitHub

### Bella

- **Specialization**: Data visualization, analysis, and interpretation.
- **Expertise**: Tableau, Excel, PowerPoint

### John

- **Specialization**: Takes charge of early report components.
- **Expertise**: Python, Excel, Jupyter Notebook

### Nathan

- **Specialization**: Responsible for data wrangling and cleaning tasks.
- **Expertise**: SAS

## Collaboration

- Collaboratively brainstorm during the analysis phase.

For more information and updates, refer to our private GitHub repository. If you have any questions or would like to discuss further aspects of the project, feel free to reach out.

---

_Last updated: September 30, 2023_
