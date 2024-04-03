#if running mac will also have to install:
#brew install libomp

# Importing Required Library
import pandas as pd

# Similarly LGBMRegressor can also be imported for a regression model.
from lightgbm import LGBMClassifier


# Reading the train and test dataset
data = pd.read_csv("/Users/nathanzlomke/Documents/GitHub/ban5753-fall2023-rxdata/Humana-Mays Competition/cleaned_humana_inner9.29")

data.head()

# Removing Columns not Required
data = data.drop(columns = ['Unnamed: 0'], axis = 1)
data.head()
#consolidating dataframe into necessary columns...
#pd.set_option('display.max_columns', None)
data.head()
model_data = data[['therapy_id', 'Visit_Duration', 'MedProcess_Duration', 'pot', 'util_cat', 'hedis_pot', 'clm_type_x', 
                   'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis', 'nausea_diagnosis', 
                   'hyperglycemia_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis','PD_C00ÐD49_FLG', 'PD_D50ÐD89_FLG',	
                   'PD_E00ÐE90_FLG',	'PD_G00ÐG99_FLG',	'PD_I00ÐI99_FLG',	'PD_J00ÐJ99_FLG',	'PD_M00ÐM99_FLG',	'PD_R00ÐR99_FLG',	
                   'PD_Z00ÐZ99_FLG',	'PD_Other_FLG',	'NPD_C00ÐD49',	'NPD_D50ÐD89', 'NPD_E00ÐE90',	'NPD_G00ÐG99',	'NPD_I00ÐI99',	
                   'NPD_J00ÐJ99',	'NPD_K00ÐK93',	'NPD_M00ÐM99',	'NPD_R00ÐR99',	'NPD_Z00ÐZ99',	'NPD_Other',	'NPD_SUM', 
                   'Prescription_Filled_Duration', 'RX_Process_Duration', 'pay_day_supply_cnt', 'rx_cost', 'tot_drug_cost_accum_amt', 
                   'mail_order_iNod', 'generic_ind', 'maint_ind', 'metric_strength', 'clm_type_y', 'ddi_ind', 'anticoag_ind',	
                   'diarrhea_treat_ind',	'nausea_treat_ind',	'seizure_treat_ind',	'RxGroupings', 'Date Duration', 'tgt_ade_dc_ind', 
                   'race_cd', 'est_age', 'sex_cd', 'cms_disabled_ind', 'cms_low_income_ind']] 
model_data.head()
#have to have all data in float, int, or bool
##remove or dummy code strings
#drop 'pot', 'util_cat', 'hedis_pot', 'clm_typ_x', 'clm_type_y'
race = pd.get_dummies(model_data['race_cd'], dtype=int, drop_first=False)
ade_flag = pd.get_dummies(model_data['ade_diagnosis'], dtype=int, drop_first=True)
seizure_flag = pd.get_dummies(model_data['seizure_diagnosis'], dtype=int, drop_first=True)
pain_flag = pd.get_dummies(model_data['pain_diagnosis'], dtype=int, drop_first=True)
mail_flag =  pd.get_dummies(model_data['mail_order_iNod'], dtype=int, drop_first=True)
generic_flag = model_data['generic_ind'].replace({'GENERIC': 1, 'BRANDED': 0})
maintenance_flag = model_data['maint_id'].replace({'MAINT': 1, 'NONMAINT': 0})
ddi_flag =  pd.get_dummies(model_data['ddi_ind'], dtype=int, drop_first=True)

#data['tgt_ade_dc_ind'] = data['tgt_ade_dc_ind'].replace({'Yes': 1, 'No': 0})
# INSERT HOLDOUT DATA HERE

# Skipping Data Exploration
# Dummification of Diagnosis Column --should we use 

data['race_cd']= pd.get_dummies(data['race_cd'])


# Splitting Dataset in two parts for testing
train = model_data[0:400]
test = model_data[400:568]

#FOR REAL
###test = pd.read_csv("HOLDOUT_MERGED_.csv")



# Separating the independent and target variable on both data set
x_train = train.drop(columns =['tgt_ade_dc_ind'], axis = 1)
y_train = train['tgt_ade_dc_ind']
x_test = test.drop(columns =['tgt_ade_dc_ind'], axis = 1)
y_test = test['tgt_ade_dc_ind']


# Creating an object for model and fitting it on training data set
model = LGBMClassifier()
model.fit(x_train, y_train)

# Predicting the Target variable
pred = model.predict(x_test)
print(pred)
accuracy = model.score(x_test, y_test)
print(accuracy)

