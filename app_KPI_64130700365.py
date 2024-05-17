import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/Uncleaned_employees_final_dataset.csv')

# Display the first few rows of the dataset to understand its structure
st.write(df.head())

# Drop 'employee_id' and target column 'KPIs_met_more_than_80' from features
X = df.drop(columns=['employee_id', 'KPIs_met_more_than_80'])
y = df['KPIs_met_more_than_80']  # Target variable

# Encode categorical features if necessary
categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Define the pipeline
model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Normalization step
    ('classifier', SVC())  # SVC classifier
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['linear', 'rbf']
}

# Create GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # You can adjust cv (cross-validation) as needed

# Fit the pipeline with GridSearchCV
grid_search.fit(X, y)

# Access the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Display the best parameters and best estimator
st.write("Best Parameters:", best_params)
st.write("Best Estimator:", best_estimator)

# Example input data for prediction
x_new = pd.DataFrame({
    'department': ['Technology'],
    'region': ['region_26'],
    'education': ['Bachelors'],
    'gender': ['m'],
    'recruitment_channel': ['sourcing'],
    'no_of_trainings': [1],
    'age': [24],
    'previous_year_rating': [2],
    'length_of_service': [1],
    'awards_won': [0],
    'avg_training_score': [77]
})

# Function to replace unseen values with the first class in the encoder
def replace_unseen_values(column, encoder):
    unseen_values = set(column) - set(encoder.classes_)
    column = column.apply(lambda x: encoder.classes_[0] if x in unseen_values else x)
    return encoder.transform(column)

# Replace unseen values with the first class in the encoder
for col in categorical_columns:
    le = LabelEncoder()
    x_new[col] = replace_unseen_values(x_new[col], le)

# Make predictions using the best estimator on the new data
y_pred_new = best_estimator.predict(x_new)

# Display the prediction result
st.write('KPIs_met_more_than_80:', y_pred_new)

# Make predictions using the best estimator on the entire dataset
y_pred = best_estimator.predict(X)

summary_eval = classification_report(y, y_pred, digits=4)
st.write(summary_eval)

# Calculate the confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
# Save the plot as an image file (e.g., PNG)
plt.savefig('confusion_matrix_xxx.png')

st.pyplot(plt)
