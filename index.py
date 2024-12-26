import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify,render_template
import json 

data = pd.read_csv("updated_accurate_disease_data.csv")

label_encoders = {}
for col in ['Sex', 'S1', 'S2', 'S3', 'Disease']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

X = data[['Age', 'Sex', 'S1', 'S2', 'S3']]
y = data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "disease_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

app = Flask(__name__)

model = joblib.load("disease_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('Home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        required_fields = ['Age', 'Sex', 'S1', 'S2', 'S3']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400


        input_data = pd.DataFrame([data])


        input_data['Sex'] = label_encoders['Sex'].transform([data['Sex']])[0]
        input_data['S1'] = label_encoders['S1'].transform([data['S1']])[0]
        input_data['S2'] = label_encoders['S2'].transform([data['S2']])[0]
        input_data['S3'] = label_encoders['S3'].transform([data['S3']])[0]

        input_symptoms = {input_data['S1'].values[0], input_data['S2'].values[0], input_data['S3'].values[0]}
        
        symptom_matches = []
        for _, row in X.iterrows():
            dataset_symptoms = {row['S1'], row['S2'], row['S3']}
            if input_symptoms == dataset_symptoms:  
                matched_row = row
                symptom_matches.append(matched_row)
                break  

        if not symptom_matches:
            symptom_matches = []
            for _, row in X.iterrows():
                dataset_symptoms = {row['S1'], row['S2'], row['S3']}
                match_count = len(input_symptoms.intersection(dataset_symptoms))
                if match_count > 0:  
                    symptom_matches.append((row, match_count))

            if symptom_matches:
                symptom_matches.sort(key=lambda x: x[1], reverse=True)  
                matched_row = symptom_matches[0][0]  

        if not symptom_matches:
            return jsonify({
                "prediction": None,
                "warning": "The provided symptoms do not match any disease in the training data."
            })

        input_data['S1'] = matched_row['S1']
        input_data['S2'] = matched_row['S2']
        input_data['S3'] = matched_row['S3']

        prediction = model.predict(input_data[['Age', 'Sex', 'S1', 'S2', 'S3']])[0]

        predicted_disease = label_encoders['Disease'].inverse_transform([prediction])[0]

        return jsonify({
            "prediction": predicted_disease,
            "warning": "Prediction based on best symptom match."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/disease/<name>', methods=['GET'])
def get_disease_info(name):
    with open("index.json", "r") as file:
        diseases_data = json.load(file)
    disease = diseases_data.get(name)
    if disease:
        return jsonify(disease)
    else:
        return jsonify({"error": "Disease not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)