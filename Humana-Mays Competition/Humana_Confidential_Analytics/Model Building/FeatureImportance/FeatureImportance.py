import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

def load_and_preprocess_data():
    # Load the datasets
    score_df = pd.read_csv("/Users/brocktbennett/2023CaseCompetition_Isabella_Lieberman_20231005.csv")

    # Dataframes to compare with score_df
    dataframes = {
        'medclms_holdout': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/medclms_holdout.csv"),
        'medclms_train': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/medclms_train.csv"),
        'rxclms_holdout': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/rxclms_holdout.csv"),
        'rxclms_train': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/rxclms_train.csv"),
        'target_holdout': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/target_holdout.csv"),
        'target_train': pd.read_csv("/Users/brocktbennett/2023_TAMU_competition_data/target_train.csv")
    }

    # Convert the ID columns to string for consistency
    score_df['ID'] = score_df['ID'].astype(str)
    print("Number of rows in score_df:", len(score_df))

    # Initialize merged_df as score_df
    merged_df = score_df

    for name, df in dataframes.items():
        id_col = None
        if 'ID' in df.columns:
            id_col = 'ID'
        elif 'id' in df.columns:
            id_col = 'id'

        if id_col:
            if id_col not in merged_df.columns:
                print(f"'{id_col}' not found in merged_df. Skipping merging with {name}.")
                continue

            df[id_col] = df[id_col].astype(str)
            temp_merged = pd.merge(merged_df, df, on=id_col, how='left', suffixes=('', f'_from_{name}'))

            if len(temp_merged) == 0:
                print(f"No common IDs between merged_df and {name}. Skipping this merge.")
                continue

            merged_df = temp_merged
            print(f"Number of rows in merged_df after merging with {name} using {id_col}:", len(merged_df))
        else:
            print(f"{name} does not have an 'ID' or 'id' column.")

    # Save the merged dataframe to a new CSV file
    save_path = "/Users/brocktbennett/post_model_eda4real.csv"
    merged_df.to_csv(save_path, index=False)  # set index=False to avoid writing row numbers
    print(f"Data saved to {save_path}")

    return merged_df

# Call the function
load_and_preprocess_data()
