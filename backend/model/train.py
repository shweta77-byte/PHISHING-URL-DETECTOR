import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Starting training...")

# Load dataset
df = pd.read_csv("data/dataset.csv")
print("Dataset loaded successfully!")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

print("Splitting data...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Evaluating model...")

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
with open("backend/model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")