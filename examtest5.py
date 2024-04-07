# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Read data from Excel file
file_path = r'C:\Users\hanna\OneDrive\College Files\Second Semester\ITSCM 180\baseball.xlsx'
data = pd.read_excel(file_path)

# Step 2: Extract relevant columns
# Columns: Runs Scored (D), Runs Allowed (E), Wins (F), OBP (G), SLG (H), Team Batting Average (I), Playoffs (J)
features = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
target = data['Playoffs']  # The variable we want to predict

# Step 3: Prepare data for training and testing
# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
# Predict on the testing set
predictions = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Step 6: Print the prediction model
# Coefficients of the model
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print("Prediction Model:")
print(f"Intercept: {intercept}")
for feature, coef in zip(features.columns, coefficients):
    print(f"{feature}: {coef}")

# Step 7: Compare predicted results to actual results
print("\nComparison of Predicted Results to Actual Results:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(comparison)
print(f"\nAccuracy of the model: {accuracy}")