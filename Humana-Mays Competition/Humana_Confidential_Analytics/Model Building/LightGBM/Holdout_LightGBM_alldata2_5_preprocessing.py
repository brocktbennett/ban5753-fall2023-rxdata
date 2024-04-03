import pandas as pd
from lightgbm import LGBMClassifier
import re
from sklearn.model_selection import RandomizedSearchCV

# Set display options
pd.set_option('display.width', 1000, 'display.max_colwidth', 100, 'display.max_columns', None)

# Load data
data_merged = pd.read_csv('/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data/holdout_cleaned/holdout_cleaned_humana_inner10-1.csv')

# Initial print
print(f"Number of Columns before drop: {data_merged.shape[1]}")
print(f"Number of Rows before drop: {data_merged.shape[0]}\n")
print("Data Types of Each Column:\n", data_merged.dtypes.to_string())
print("\nTop 10 Rows of DataFrame:\n", data_merged.head(10))

# Data preprocessing
data_merged.drop(columns=['Unnamed: 0'], inplace=True)
data_merged.rename(columns= {'RX_Groupings': 'RxGroupings'}, inplace=True)
model_data = data_merged['therapy_id', 'Visit_Duration', 'MedProcess_Duration', 'pot', 'util_cat', 'hedis_pot', 'clm_type_x',
     'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis', 'nausea_diagnosis',
     'hyperglycemia_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis', 'PD_C00ÐD49_FLG',
     'PD_D50ÐD89_FLG', 'PD_E00ÐE90_FLG', 'PD_G00ÐG99_FLG', 'PD_I00ÐI99_FLG', 'PD_J00ÐJ99_FLG',
     'PD_M00ÐM99_FLG', 'PD_R00ÐR99_FLG', 'PD_Z00ÐZ99_FLG', 'PD_Other_FLG', 'NPD_C00ÐD49',
     'NPD_D50ÐD89', 'NPD_E00ÐE90', 'NPD_G00ÐG99', 'NPD_I00ÐI99', 'NPD_J00ÐJ99', 'NPD_K00ÐK93',
     'NPD_M00ÐM99', 'NPD_R00ÐR99', 'NPD_Z00ÐZ99', 'NPD_Other', 'NPD_SUM', 'Prescription_Filled_Duration',
     'RX_Process_Duration', 'pay_day_supply_cnt', 'rx_cost', 'tot_drug_cost_accum_amt', 'mail_order_iNod',
     'generic_ind', 'maint_ind', 'metric_strength', 'clm_type_y', 'clm_type_x', 'ddi_ind', 'anticoag_ind',
     'diarrhea_treat_ind', 'nausea_treat_ind', 'seizure_treat_ind', 'RxGroupings',
     'race_cd', 'est_age', 'sex_cd', 'cms_disabled_ind', 'cms_low_income_ind']
binary_columns = [
    'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis',
    'nausea_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis',
    'hyperglycemia_diagnosis', 'ddi_ind', 'anticoag_ind', 'diarrhea_treat_ind',
    'nausea_treat_ind', 'seizure_treat_ind', 'cms_disabled_ind', 'cms_low_income_ind']

for col in binary_columns:
    model_data.loc[:, col] = model_data[col].replace({'Yes': 1, 'No': 0}).fillna(value=3).astype(int)

model_data['maint_ind'] = model_data['maint_ind'].replace({'MAINT': 1, 'NONMAINT': 0})
model_data['generic_ind'] = model_data['generic_ind'].replace({'GENERIC': 1, 'BRANDED': 0})
model_data.drop('generic_ind', axis=1, inplace=True)
model_data = pd.get_dummies(model_data, columns=['clm_type_y', 'mail_order_iNod'], dtype=int)
model_data['sex_cd'] = model_data['sex_cd'].replace({'M': 1, 'F': 0}).fillna(value=3).astype(int)
model_data['est_age'] = model_data['est_age'].fillna(value=(model_data['est_age'].mean())).astype(int)
model_data['metric_strength'] = model_data['metric_strength'].fillna(value=0).astype(int)

# Dummies
RX_GroupDF = pd.get_dummies(model_data['RxGroupings'], prefix='RXGrouping', dtype=int)
Race_DF = pd.get_dummies(model_data['race_cd'], prefix='race', dtype=int)
util_DF = pd.get_dummies(model_data['util_cat'], prefix='util_cat_', dtype=int)
model_data.drop(columns=['race_cd', 'RxGroupings', 'util_cat'], inplace=True)
model_data = pd.concat([model_data, Race_DF, RX_GroupDF, util_DF], axis=1)

# Clean therapy_id
cleaned_id = [re.sub(r'([-])\w+', '', item) for item in model_data['therapy_id']]
model_data['cleaned_id'] = pd.Series(cleaned_id).astype(int)
model_data.drop(columns=['therapy_id', 'clm_type_x', 'pot', 'hedis_pot'], inplace=True)

print(model_data)

model_data.info()
path = "/Users/brocktbennett/GitHub/Project Data/Holdout_Data/CLEAN_HOLDOUT_10-3.csv"
model_data.to_csv(path, index=False)

# Model Training
train = pd.read_csv("/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data/holdout_cleaned/model_data10-2.csv")
train.drop(columns={'Date Duration'}, inplace=True)
test = model_data
test[['race_North American Native', 'util_cat__IP_LTACH', 'util_cat__IP_MHSA']] = 0

x_train = train.drop(columns=['tgt_ade_dc_ind'])
y_train = train['tgt_ade_dc_ind']
x_test = test
y_test = data_merged['y_test']

# Hyperparameter tuning
lgbm_params = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "num_leaves": [8, 12, 24, 32]
}

rand = RandomizedSearchCV(LGBMClassifier(seed=0), lgbm_params, n_iter=10, cv=3, scoring='roc_auc', random_state=1)
rand.fit(x_train, y_train)

# Train the best model
model = LGBMClassifier(**rand.best_params_, seed=0)
model.fit(x_train, y_train)

# Predictions
pred = model.predict(x_test)
print(pred)

# Evaluate model
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
