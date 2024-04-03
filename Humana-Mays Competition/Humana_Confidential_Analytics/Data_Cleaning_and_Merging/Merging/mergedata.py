# Import necessary libraries
import pandas as pd


# Define a function to load data
def load_data(base_path):
    datasets = {}
    # List of filenames to load
    filenames = ['target_train', 'medclms_train', 'rxclms_train']

    for filename in filenames:
        dataset = pd.read_csv(f"{base_path}/{filename}.csv", low_memory=False)
        dataset.fillna('N/A', inplace=True)
        datasets[filename] = dataset

    return datasets


# Define the base path where the data is located
base_path = "/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data"

# Load the datasets using the defined function
datasets = load_data(base_path)

# # Merge all datasets together on 'therapy_id' as the primary key using a full outer join
# merged_data = datasets['medclms_train'].merge(datasets['rxclms_train'], on='therapy_id', how='outer')
# merged_data = merged_data.merge(datasets['target_train'], on='therapy_id', how='outer')

# Merge all datasets together on 'therapy_id' as the primary key using an inner join
merged_data = datasets['medclms_train'].merge(datasets['rxclms_train'], on='therapy_id', how='inner')
merged_data = merged_data.merge(datasets['target_train'], on='therapy_id', how='inner')


# Get the first few rows of the merged dataset (e.g., the first 5 rows)
print("First few rows of the merged dataset:")
print(merged_data.head())

# Print the number of rows and columns in the merged dataset
num_rows, num_columns = merged_data.shape
print(f"This merged dataset has {num_rows} rows and {num_columns} columns.")

# Get basic statistics of the merged dataset
print("\nBasic statistics of the merged dataset:")
print(merged_data.describe())

# # Export the merged DataFrame to a new CSV file in the specified directory
# merged_data.to_csv("/Users/brocktbennett/GitHub/Project Data/2023_TAMU_competition_data/Merged/cleaned_humana_inner.csv", index=False)
