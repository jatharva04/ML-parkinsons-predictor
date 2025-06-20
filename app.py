import joblib
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained SVM model and scaler
model = joblib.load('svm_model.pkl') # to predict yes/no
scaler_svm = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl') # to predict severity score
model_classification = joblib.load('rf_classifier.pkl') # to categorize severity into levels
scaler_rf = joblib.load('scaler_rf.pkl')


def classify_severity(total_updrs):
    """Classify the severity of Parkinson's disease based on UPDRS score."""
    if total_updrs <= 20:
        return 'Mild', '0-20'
    elif 21 <= total_updrs <= 40:
        return 'Moderate', '21-40'
    else:
        return 'Severe', '41+'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['MDVP_Fo']),
        float(request.form['MDVP_Fhi']),
        float(request.form['MDVP_Flo']),
        float(request.form['MDVP_Jitter_pc']),
        float(request.form['MDVP_Jitter_abs']),
        float(request.form['MDVP_RAP']),
        float(request.form['MDVP_PPQ']),
        float(request.form['Jitter_DDP']),
        float(request.form['MDVP_Shimmer']),
        float(request.form['MDVP_Shimmer_db']),
        float(request.form['Shimmer_APQ3']),
        float(request.form['Shimmer_APQ5']),
        float(request.form['MDVP_APQ']),
        float(request.form['Shimmer_DDA']),
        float(request.form['NHR']),
        float(request.form['HNR']),
        float(request.form['RPDE']),
        float(request.form['DFA']),
        float(request.form['spread1']),
        float(request.form['spread2']),
        float(request.form['D2']),
        float(request.form['PPE']),
    ]
    
    # Convert inputs to numpy array and scale it
    input_data = np.asarray(input_data).reshape(1, -1)
    scaled_data_svm = scaler_svm.transform(input_data)
    
    # Predict using the trained SVM model
    disease_prediction = model.predict(scaled_data_svm)
    if disease_prediction[0] == 0:
        result_disease = 'Negative'
        severity_label, limit, updrs_score = 'N/A', 'N/A', None
    else:
        result_disease = 'Positive'
        
        updrs_input_data = np.asarray([
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['test_time']),
            float(request.form['motor_UPDRS']),
            float(request.form['MDVP_Jitter_pc']),
            float(request.form['MDVP_Jitter_abs']),
            float(request.form['MDVP_RAP']),
            float(request.form['MDVP_PPQ']),
            float(request.form['Jitter_DDP']),
            float(request.form['MDVP_Shimmer']),
            float(request.form['MDVP_Shimmer_db']),
            float(request.form['Shimmer_APQ3']),
            float(request.form['Shimmer_APQ5']),
            float(request.form['Shimmer_APQ11']),
            float(request.form['Shimmer_DDA']),
            float(request.form['NHR']),
            float(request.form['HNR']),
            float(request.form['RPDE']),
            float(request.form['DFA']),
            float(request.form['PPE']),
        ]).reshape(1, -1)
        
        scaled_data_rf = scaler_rf.transform(updrs_input_data)
        updrs_score = rf_model.predict(scaled_data_rf)[0]
        
        severity_input_data = np.array([
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['test_time']),
            float(request.form['motor_UPDRS']),
            float(request.form['MDVP_Jitter_pc']),
            float(request.form['MDVP_Jitter_abs']),
            float(request.form['MDVP_RAP']),
            float(request.form['MDVP_PPQ']),
            float(request.form['Jitter_DDP']),
            float(request.form['MDVP_Shimmer']),
            float(request.form['MDVP_Shimmer_db']),
            float(request.form['Shimmer_APQ3']),
            float(request.form['Shimmer_APQ5']),
            float(request.form['Shimmer_APQ11']),
            float(request.form['Shimmer_DDA']),
            float(request.form['NHR']),
            float(request.form['HNR']),
            float(request.form['RPDE']),
            float(request.form['DFA']),
            float(request.form['PPE']),
        ])

        # Reshape severity_input_data to (1, 20)
        severity_input_data = severity_input_data.reshape(1, -1)

        # Debugging line to check shape
        print(f"Severity Input Data Shape: {severity_input_data.shape}")

        # Make prediction using the classification model
        severity_prediction = model_classification.predict(severity_input_data)

        severity_prediction = model_classification.predict(severity_input_data)
        severity_label, limit = classify_severity(updrs_score)

    return render_template('form.html', result=result_disease, severity=severity_label, limit=limit, updrs_score=updrs_score)

if __name__ == '__main__':
    app.run(debug=True)
