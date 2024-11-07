from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

app = Flask(__name__)

# Load the pre-fitted StandardScaler
pipeline = load('attrition_pipeline.joblib')
rfc = pipeline['model']
ss = pipeline['scaler']
label_encoders = pipeline['encoders']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mapping for categorical feature 'BusinessTravel'
    business_travel_map = {
        'Travel_Rarely': 1.0,
        'Travel_Frequently': 2.0,
        'Non-Travel': 0.0
    }

    # Retrieve form inputs and map categorical fields
    data = {
        'Age': float(request.form['Age']),
        'BusinessTravel': business_travel_map.get(request.form['BusinessTravel'], 0.0),
        'JobSatisfaction': float(request.form['JobSatisfaction']),
        'MonthlyIncome': float(request.form['MonthlyIncome']),
        'WorkLifeBalance': float(request.form['WorkLifeBalance']),
        'YearsAtCompany': float(request.form['YearsAtCompany']),
        'YearsInCurrentRole': float(request.form['YearsInCurrentRole'])
    }

    # Convert to DataFrame for transformation
    df = pd.DataFrame([data])

    # Scale features using the loaded scaler
    features = ss.transform(df)

    # Generate prediction and probability
    prediction = rfc.predict(features)[0]  # Prediction (class label)
    probability = rfc.predict_proba(features)[0][prediction]  # Probability of the predicted class

    # Render the result template
    return render_template('result.html', prediction=prediction, probability=round(probability, 2))

if __name__ == '__main__':
    app.run(debug=True)
