# ============================
# 1. DATA LOADING
# ============================

import pandas as pd


def load_data(base_path):
    datasets = {}
    for filename in ['data_dictionary', 'target_holdout', 'target_train',
                     'medclms_holdout', 'medclms_train', 'rxclms_holdout', 'rxclms_train',
                     'race_cd_desc']:
        dataset = pd.read_csv(f"{base_path}/{filename}.csv", header=None, low_memory=False)
        dataset.fillna('N/A', inplace=True)
        # dataset = convert_dates(dataset)  # Convert dates
        datasets[filename] = dataset
    return datasets


# def convert_dates(datasets):
#     date_format = '%Y-%m-%dT%H:%M:%S.%fZ'  # specify the date format
#     for col in datasets.columns:  # loop through each column in the dataset
#         # temporarily store the original data in case the conversion fails
#         original_data = datasets[col].copy()
#         try:
#             # try to convert the column to datetime
#             datasets[col] = pd.to_datetime(datasets[col], format=date_format, errors='coerce')
#             # if conversion is successful, format the datetime as a string
#             datasets[col] = datasets[col].apply(lambda x: x.strftime('%m/%d/%Y') if not pd.isnull(x) else x)
#             # check if the conversion was successful by checking if any dates were successfully converted
#             if datasets[col].isnull().all():
#                 # if all values are NaT (not a time), revert the changes
#                 datasets[col] = original_data
#         except Exception as e:
#             # if any other exception occurs, revert the changes
#             datasets[col] = original_data
#             print(f"An error occurred while converting column '{col}': {e}")
#     return datasets

def convert_dates(datasets):
    date_format = '%Y-%m-%dT%H:%M:%S.%fZ'  # specify the date format
    for col in datasets.columns:  # loop through each column in the dataset
        if datasets[col].dtype == 'object':
            try:
                # try to convert the column to datetime
                datasets[col] = pd.to_datetime(datasets[col], format=date_format, errors='coerce')
                # if conversion is successful, format the datetime as a day-month-year string
                datasets[col + '_human_readable'] = datasets[col].dt.strftime('%d %B %Y')
            except Exception as e:
                print(f"An error occurred while converting column '{col}': {e}")
    return datasets


base_path = "/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data"
datasets = load_data(base_path)

# Set display options outside the loop
pd.set_option('display.width', 1000, 'display.max_colwidth', 100, 'display.max_columns', None)

# Displaying the details for each dataset
for name, dataset in datasets.items():
    print(f"\nDataset Name: {name}")
    print(f"Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(f"Head: {dataset.head()}")
    print("-" * 50)

    # Generating and displaying the statistical summary
    description = dataset.describe(include='all')  # include='all' to describe all columns
    print(f"Statistical Summary: \n{description}\n")
    print("-" * 50)

    # If you need to convert the statistical summary to a dictionary
    description_dict = description.to_dict()

# ============================
# 2. DATA CLEANING
# ============================

# Handle 'null' and 'NaN' values
for dataset in datasets.values():
    dataset.fillna(0, inplace=True)  # Filling NaN values with 0 as an example
    # For more advanced handling, you could use:
    # dataset.fillna(dataset.mean(), inplace=True)  # Fills with mean of columns

# ============================
# 3. DATA MASKING
# ============================


# # ============================
# # 4. FEATURE ENGINEERING
# # ============================
# # Add your feature engineering code here
#
# # ============================
# # 5. MODEL BUILDING & EVALUATION
# # ============================
#
# X = datasets['target_train'][["feature_1", "feature_2"]]
# y = datasets['target_train']["outcome"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
#
# # ============================
# # 6. OUTPUT PREPARATION
# # ============================
#
# output_df = datasets['target_train'].copy()
#
# for col in sensitive_columns:
#     if col in output_df.columns:
#         output_df[col] = "MASKED"
#
# output_df.to_csv("output_masked.csv", index=False)
