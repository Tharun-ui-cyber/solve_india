from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        attendance = float(request.form['attendance'])
        marks = float(request.form['marks'])
        study_hours = float(request.form['study_hours'])
        
        # Predict dropout risk
        features = np.array([[attendance, marks, study_hours]])
        prediction_value = model.predict(features)[0]
        
        if prediction_value == 1:
            prediction = "High Risk of Dropout"
            risk_label = 1
        else:
            prediction = "Low Risk of Dropout (Safe)"
            risk_label = 0
            
        # Save prediction to CSV file
        import csv
        csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_student_data.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['attendance', 'marks', 'study_hours', 'dropout_risk'])
            writer.writerow([attendance, marks, study_hours, risk_label])
            
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
