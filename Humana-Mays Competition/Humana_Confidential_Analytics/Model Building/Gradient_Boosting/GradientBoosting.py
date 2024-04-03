# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Set display options
pd.set_option('display.width', 1000, 'display.max_colwidth', 100, 'display.max_columns', None)

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

# Merge all datasets together on 'therapy_id' as the primary key using an inner join
merged_data = datasets['medclms_train'].merge(datasets['rxclms_train'], on='therapy_id', how='inner')
merged_data = merged_data.merge(datasets['target_train'], on='therapy_id', how='inner')

# Convert 'Yes' to 1 and 'No' to 0 in the 'tgt_ade_dc_ind' column
merged_data['tgt_ade_dc_ind'] = merged_data['tgt_ade_dc_ind'].replace({'Yes': 1, 'No': 0})

# Exclude non-numeric columns from PCA (you can adjust this based on your dataset)
X_numeric = merged_data.select_dtypes(include=[float, int])

# Prepare features (X) and target (y)
X = X_numeric.drop(columns=['tgt_ade_dc_ind'])  # Exclude the target column from the features
y = merged_data['tgt_ade_dc_ind']  # Define the target column

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality (you can adjust the number of components)
pca = PCA(n_components=10)  # Adjust the number of components as needed
X_pca = pca.fit_transform(X_scaled)

# Split the PCA-transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create and train the gradient boosting classifier with hyperparameter tuning
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # You can adjust these values
    'learning_rate': [0.01, 0.1, 0.2],  # You can adjust these values
    'max_depth': [3, 4, 5],  # You can adjust these values
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_gb_classifier = grid_search.best_estimator_

# Make predictions using the best classifier
y_pred = best_gb_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Calculate AUC (Area Under the ROC Curve)
y_pred_proba = best_gb_classifier.predict_proba(X_test)[:, 1]  # Probabilities of positive class
auc_score = roc_auc_score(y_test, y_pred_proba)

# Print performance metrics
print("Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print("AUC Score:", auc_score)

# Access the principal components
components = pca.components_

# Visualize explained variance ratios for principal components
explained_variance_ratios = pca.explained_variance_ratio_
component_indices = np.arange(1, len(explained_variance_ratios) + 1)

# Create the bar chart
plt.bar(component_indices, explained_variance_ratios, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratios for Principal Components')
plt.xticks(component_indices)
plt.show()

# Interpret the components and associate with feature names
for i, component in enumerate(components):
    print(f"Principal Component {i + 1}:")
    for feature_name, feature_weight in zip(X.columns, component):
        print(f"   {feature_name}: {feature_weight}")
