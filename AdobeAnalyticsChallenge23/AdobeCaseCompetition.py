# ========================================================================
# Adobe Case Competition
# Team: Stone_cloud_Masters, Team 13
# Representing: Oklahoma State University
# Master of Science in Business Analytics and Data Science
# Team Members:
#   - Isabella Lieberman (Team Captain)
#   - Brock Bennett
#   - John Ramirez
# ====================================================================
# John added
# Possibly place our data in a database so we do not need to worry about it...

# [1] Import Necessary Libraries
#   - pandas: Data manipulation
#   - numpy: Numeric calculations
#   - sklearn: Machine learning algorithms and utilities
import numpy as pd
import sklearn
# etc.

# [2] Data Collection Phase
# ------------------------------------------------
#   A. Load Data
df = pd.read_csv('dataset.csv')

#   B. Data Review
print(df.head())
print(df.describe())
print(df.info())

# [3] Data Cleaning & Preprocessing
# ------------------------------------------------
#   A. Null Value Handling
df.fillna(0, inplace=True)

#   B. Categorial Data Conversion
pd.get_dummies(df['some_column'])

#   C. Feature Scaling
#   (Your code here)

#   D. Feature Selection
#   (Your code here)

# [4] Ensure Data Privacy & Compliance (HIPAA)
# ------------------------------------------------
#   A. Drop Sensitive Information
df.drop(columns=['Name', 'SocialSecurityNumber', 'Address'], inplace=True)

#   B. Data Anonymization
df['Age'] = df['Age'].apply(lambda x: x + np.random.randint(-5, 5))

#   C. Data Encryption
#   (Your code here)

# [5] Partition Data into Training & Testing Sets
# ------------------------------------------------
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# [6] Model Creation and Training
# ------------------------------------------------
#   A. Model Initialization
model = sklearn.ensemble.RandomForestClassifier()

#   B. Model Training
model.fit(X_train, y_train)

# [7] Model Assessment
# ------------------------------------------------
#   A. Prediction on Test Data
y_predicted = model.predict(X_test)

#   B. Evaluate Model Metrics
print(sklearn.metrics.accuracy_score(y_test, y_predicted))
print(sklearn.metrics.confusion_matrix(y_test, y_predicted))

# [8] Deploy Model (if applicable)
# ------------------------------------------------
#   (Your code here)

# [9] Generate Insights and Report
# ------------------------------------------------
#   (Your code here)
