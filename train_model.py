import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Define file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'student_data.csv')
model_path = os.path.join(base_dir, 'model.pkl')

# Load dataset
df = pd.read_csv(data_path)

# Features and Target
X = df[['attendance', 'marks', 'study_hours']]
y = df['dropout']

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model as model.pkl
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Model trained successfully!")
print(f"Saved to: {model_path}")
