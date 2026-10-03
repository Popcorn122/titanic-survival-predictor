# 1. Import the tools we need
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 3. Select the features (inputs) and target (what we want to predict)
features = ['Pclass', 'Sex', 'Age', 'Fare']
X = data[features].copy()
y = data['Survived']

# Convert 'Sex' from text to numbers (0 for male, 1 for female)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Fill in missing ages with the average age
X['Age'] = X['Age'].fillna(X['Age'].mean())

# 4. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create the AI Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Train the model
print("Training the AI model...")
model.fit(X_train, y_train)

# 7. Test the model and see how accurate it is
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"AI Model Accuracy: {accuracy * 100:.2f}%")
