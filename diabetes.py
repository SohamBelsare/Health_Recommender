import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    return pd.read_csv('diabetes_prediction_dataset.csv')

def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])
    return data

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

def make_predictions(model, new_data):
    new_prediction = model.predict_proba(new_data)[:, 1]
    
    print(f'Probability of having diabetes: {new_prediction[0]:.2%}')
    print(f'Prediction: {"Yes" if new_prediction[0] >= 0.5 else "No"}')

def main():
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    data = preprocess_data(data)
    X = data[['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    new_data = pd.DataFrame({'gender': [0], 'age': [30], 'hypertension': [0], 'heart_disease': [0], 'bmi': [25], 'HbA1c_level': [5.5], 'blood_glucose_level': [90]})
    make_predictions(model, new_data)
main()

if __name__ == "__main__":
    main()