from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import requests

app = Flask(__name__)

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(['Person ID','Occupation','Quality of Sleep','Blood Pressure'], axis=1)
label_encoder = preprocessing.LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])

X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

@app.route('/predict', methods=['GET'])
def predict():
    return "connected"
    # if request.method == 'POST':
    #     data = request.get_json()
    #     gender = data['Gender']
    #     age = data['Age']
    #     family_history = data['Family History']
    #     exercise = data['Exercise (hours)']
    #     diet = data['Diet Rating']
    #     alcohol = data['Alcohol Consumption (glasses)']
    #     height = data['Height']
    #     weight = data['Weight']
    #
    #     prediction = classifier.predict([[gender, age, family_history, exercise, diet, alcohol, height, weight]])
    #
    #     return jsonify({'prediction': str(prediction[0])})
    # else:
    #     return "Send a POST request to this endpoint with the required data."

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

# Sending data input to the Flask API
data = {
    "Gender": 0,
    "Age": 20,
    "Family History": 3,
    "Exercise (hours)": 30,
    "Diet Rating": 8,
    "Alcohol Consumption (glasses)": 0,
    "Height": 70,
    "Weight": 2000
}

response = requests.get('http://127.0.0.1:5000/predict', json=data)
print(response.json())
