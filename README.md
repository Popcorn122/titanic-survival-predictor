# Titanic Survival Predictor 🚢

## 📌 Project Overview
This is my first Machine Learning project. The goal is to build a model that predicts whether a passenger on the Titanic survived or not, based on features like their age, gender, ticket class, and fare.

## 🛠️ Technologies Used
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn

## 🧠 How it Works
The script (`titanic_model.py`) does the following:
1.  **Loads** the Titanic dataset directly from a URL.
2.  **Cleans** the data by filling in missing ages and converting text (Male/Female) into numbers.
3.  **Splits** the data (80% for training, 20% for testing).
4.  **Trains** a Random Forest Classifier model.
5.  **Evaluates** the model's accuracy.

## 🚀 How to Run This Project
1. Make sure you have Python installed.
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn
