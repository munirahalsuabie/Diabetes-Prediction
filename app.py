from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
import joblib

app = Flask(__name__, template_folder='template', static_folder='static')

# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Load the trained machine learning model
model = joblib.load('model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    
    return render_template('index.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
     # Get the input values from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    hypertension = request.form['hypertension']
    heart_disease = request.form['heart_disease']
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])

  # Convert categorical variables to numerical variables
    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    if hypertension == 'Yes':
        hypertension = 1
        
    else:
        hypertension = 0

    if heart_disease == 'Yes':
        heart_disease = 1
    else:
        heart_disease = 0
      


    # Use the machine learning model to make a prediction
    input_data = np.array([gender, age, blood_glucose_level, hypertension, heart_disease, bmi, HbA1c_level]).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    # Render the prediction result in same HTML page
    return render_template('index.html', prediction=prediction)
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



