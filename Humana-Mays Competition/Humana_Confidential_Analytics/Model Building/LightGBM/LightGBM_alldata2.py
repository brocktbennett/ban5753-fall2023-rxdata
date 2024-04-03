# Import necessary Libraries
import pandas as pd
from lightgbm import LGBMClassifier

# Set display options
pd.set_option('display.width', 1000, 'display.max_colwidth', 100, 'display.max_columns', None)

# Reading the train and test dataset
data_merged = pd.read_csv(
#    "/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data/Merged/New/cleaned_humana_inner9-29.csv")
    "/Users/nathanzlomke/Downloads/CORRECTED_cleaned_humana_inner10-9.csv", low_memory=False)
# Print number of columns and rows in the DataFrame
print("Print number of columns and rows in the DataFrame:")
print(f"Number of Columns before drop: {data_merged.shape[1]}")
print(f"Number of Rows before drop: {data_merged.shape[0]}\n")
initial_num_columns = data_merged.shape[1]
initial_num_rows = data_merged.shape[0]

# Print out the amount of columns and rows we are working with
# Display data types and top 10 rows with the Wine.CSV dataset.
print("Data Types of Each Column:\n", data_merged.dtypes.to_string())
print("\nTop 10 Rows of DataFrame:\n", data_merged.head(10))

# Remove Columns Not required for our analysis
data_merged = data_merged.drop(columns=['Unnamed: 0'], axis=1)


# Consolodating dataframe into necessary columns...
# 'Date Duration' was removed from this list, as it is not present in the HOLDOUT data
model_data = data_merged[
    ['therapy_id', 'Visit_Duration', 'MedProcess_Duration', 'pot', 'util_cat', 'hedis_pot', 'clm_type_x',
     'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis', 'nausea_diagnosis',
     'hyperglycemia_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis', 'PD_C00ÐD49_FLG',
     'PD_D50ÐD89_FLG', 'PD_E00ÐE90_FLG', 'PD_G00ÐG99_FLG', 'PD_I00ÐI99_FLG', 'PD_J00ÐJ99_FLG',
     'PD_M00ÐM99_FLG', 'PD_R00ÐR99_FLG', 'PD_Z00ÐZ99_FLG', 'PD_Other_FLG', 'NPD_C00ÐD49',
     'NPD_D50ÐD89', 'NPD_E00ÐE90', 'NPD_G00ÐG99', 'NPD_I00ÐI99', 'NPD_J00ÐJ99', 'NPD_K00ÐK93',
     'NPD_M00ÐM99', 'NPD_R00ÐR99', 'NPD_Z00ÐZ99', 'NPD_Other', 'NPD_SUM', 'Prescription_Filled_Duration',
     'RX_Process_Duration', 'pay_day_supply_cnt', 'rx_cost', 'tot_drug_cost_accum_amt', 'mail_order_iNod',
     'generic_ind', 'maint_ind', 'metric_strength', 'clm_type_y', 'clm_type_x', 'ddi_ind', 'anticoag_ind',
     'diarrhea_treat_ind', 'nausea_treat_ind', 'seizure_treat_ind', 'RxGroupings',
     'tgt_ade_dc_ind', 'race_cd', 'est_age', 'sex_cd', 'cms_disabled_ind', 'cms_low_income_ind']]

# Convert Categorical data back to binary for analysis
# Columns to replace 'Yes' with 1 and 'No' with 0
binary_columns = [
    'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis',
    'nausea_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis',
    'hyperglycemia_diagnosis', 'ddi_ind', 'anticoag_ind', 'diarrhea_treat_ind',
    'nausea_treat_ind', 'seizure_treat_ind', 'cms_disabled_ind', 'cms_low_income_ind'
]

# Need to convert all N/A values to 3s for third category.  
# Otherwise can't convert to numeric int.  ONLY FOR BINARY
# Apply replacements using a loop
for col in binary_columns:
    model_data.loc[:, col] = model_data[col].replace({'Yes': 1, 'No': 0})
    model_data.loc[:, col] = model_data[col].fillna(value=3)
    model_data.loc[:, col] = model_data[col].astype(int)

model_data[['ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis',
    'nausea_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis',
    'hyperglycemia_diagnosis', 'ddi_ind', 'anticoag_ind', 'diarrhea_treat_ind',
    'nausea_treat_ind', 'seizure_treat_ind', 'cms_disabled_ind', 'cms_low_income_ind']] = model_data[['ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis',
    'nausea_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis',
    'hyperglycemia_diagnosis', 'ddi_ind', 'anticoag_ind', 'diarrhea_treat_ind',
    'nausea_treat_ind', 'seizure_treat_ind', 'cms_disabled_ind', 'cms_low_income_ind']].astype(int)

# Replace 'MAINT' with 1 and 'NONMAINT' with 0 in 'maint_ind' column
model_data['maint_ind'] = model_data['maint_ind'].replace({'MAINT': 1, 'NONMAINT': 0})

#Replace 'GENERIC' with 1 and 'BRANDED' with 0
model_data['generic_ind'] = model_data['generic_ind'].replace({'GENERIC': 1, 'BRANDED': 0})
model_data = model_data.drop('generic_ind', axis=1)

# Use pd.get_dummies for 'clm_type_y' and 'mail_order_iNod' columns
model_data = pd.get_dummies(model_data, columns=['clm_type_y', 'mail_order_iNod'], dtype=int, drop_first=False)

# Replace 'M' with 1 and 'F' with 0 in 'sex_cd' column
model_data['sex_cd'] = model_data['sex_cd'].replace({'M': 1, 'F': 0})

#Third category for NA
model_data['sex_cd'] = model_data['sex_cd'].fillna(value=3)
model_data['sex_cd'] = model_data['sex_cd'].astype(int)

#Fix NA columns: 'est_age' = avg(age)
model_data['est_age'] = model_data['est_age'].fillna(value=(model_data['est_age'].mean()))
model_data['est_age'] = model_data['est_age'].astype(int)

#Fix NA columns: 'metric_strength' = 0; this is usually in cases where it was supplies (e.g., syringes)
model_data['metric_strength'] = model_data['metric_strength'].fillna(value=0)
model_data['metric_strength'] = model_data['metric_strength'].astype(int)

# Create dummy variables for 'RxGroupings', 'race_cd' and 'util_cat' columns
RX_GroupDF = pd.get_dummies(model_data['RxGroupings'], prefix='RXGrouping', dtype=int, drop_first=False)
Race_DF = pd.get_dummies(model_data['race_cd'], prefix='race', dtype=int, drop_first=False)
model_data = model_data.drop(columns=['race_cd'], axis=1)
model_data = model_data.drop(columns=['RxGroupings'], axis=1)

util_DF = pd.get_dummies(model_data['util_cat'], prefix='util_cat_', dtype=int, drop_first=False)
model_data = model_data.drop(columns=['util_cat'], axis=1)

# Display the first few rows of the 'Race_DF' DataFrame
Race_DF.head()

# Create a new DataFrame 'Dummies_df' and add 'Race_DF' and 'RX_GroupDF' as columns
Dummies_df = Race_DF
Dummies_df = Dummies_df.join(RX_GroupDF)
Dummies_df = Dummies_df.join(util_DF)
# Add 'Dummies_df' columns to the 'model_data' DataFrame
model_data = model_data.join(Dummies_df)

#Keep only the numeric left portion of therapy_id so can be numeric
import re
test_id = model_data['therapy_id']
test_id
cleaned_id = []

i=0
for item in test_id:
    cleaned_item = re.sub(r'([-])\w+', '', item)
    cleaned_id.append(cleaned_item)
cleanedID_df = pd.DataFrame(cleaned_id)

cleanedID_df[0].astype(int)
cleanedID_df= cleanedID_df.rename(columns={0:"cleaned_id"})
cleanedID_df['cleaned_id'] = cleanedID_df['cleaned_id'].astype(int)
model_data = model_data.join(cleanedID_df)
model_data = model_data.drop(columns=['therapy_id'], axis=1)

model_data = model_data.drop(columns=['clm_type_x'], axis=1)
model_data = model_data.drop(columns=['pot'], axis=1)
model_data = model_data.drop(columns=['hedis_pot'], axis=1)

#Third category for NA

#Indices 10-13
MoreBinaryCol = model_data.iloc[:,10:31].columns.tolist()
for col in MoreBinaryCol:
    model_data.loc[:, col] = model_data[col].fillna(value=3)

#maint_ind change NA to 3
model_data['maint_ind'] = model_data['maint_ind'].fillna(value=3)
model_data['maint_ind'] = model_data['maint_ind'].astype(int)

#Fix NA columns to impute mean: Visit_Duration, MedProcess_Duration, NPD_SUM, 
# Prescription_Filled_Duration, RX_Process_Duration, pay_day_supply_cnt, rx_cost,
#tot_drug_cost_accum_amt, 
ImputeMeanCol = model_data.iloc[:,31:37].columns.tolist()
ImputeMeanCol.append('Visit_Duration')
ImputeMeanCol.append('MedProcess_Duration')

for col in ImputeMeanCol:
    model_data.loc[:, col] = model_data.loc[:, col].fillna(value=(model_data.loc[:, col].mean()))
    model_data.loc[:, col] = model_data.loc[:, col].astype(int)


# Display information about the 'model_data' DataFrame
model_data.info()

pd.set_option('display.max_rows', 100)
print(model_data.isnull().any())


model_data.to_csv("/Users/nathanzlomke/Downloads/CORRECTED_model_data10-9.csv", index=0)

