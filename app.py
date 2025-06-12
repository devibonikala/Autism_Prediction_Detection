from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for production

# Load model and setup encoders
try:
    model = pickle.load(open("model/autism_prediction_model.pkl", "rb"))
    
    # Initialize and fit label encoders
    label_encoders = {
        'ethnicity': LabelEncoder(),
        'contry_of_res': LabelEncoder(),
        'relation': LabelEncoder()
    }
    
    label_encoders['ethnicity'].fit(['White-European', 'Latino', 'Asian', 'Black', 'Middle Eastern', 'Others'])
    label_encoders['contry_of_res'].fit(['United States', 'United Kingdom', 'India', 'Others'])
    label_encoders['relation'].fit(['Parent', 'Relative', 'Self', 'Health care professional', 'Others'])
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

# Simple user storage (in-memory, will reset when server restarts)
users = {}

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', 
                               error="Invalid credentials or account not found",
                               username=username,
                               show_signup_prompt=True)
    
    signup_success = request.args.get('signup_success') == 'true'
    return render_template('login.html', 
                         signup_success=signup_success,
                         username=request.args.get('username'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users:
            return render_template('signup.html', 
                                error="Username already exists",
                                username=username)
        
        users[username] = password
        return redirect(url_for('login', 
                             signup_success='true',
                             username=username))
    
    return render_template('signup.html')

@app.route('/homepage')
def homepage():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('homepage.html', username=session['username'])

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('predict.html', username=session['username'])
    
    # Handle prediction form submission
    try:
        input_data = {
            'A1_Score': int(request.form.get("A1_Score", 0)),
            'A2_Score': int(request.form.get("A2_Score", 0)),
            'A3_Score': int(request.form.get("A3_Score", 0)),
            'A4_Score': int(request.form.get("A4_Score", 0)),
            'A5_Score': int(request.form.get("A5_Score", 0)),
            'A6_Score': int(request.form.get("A6_Score", 0)),
            'A7_Score': int(request.form.get("A7_Score", 0)),
            'A8_Score': int(request.form.get("A8_Score", 0)),
            'A9_Score': int(request.form.get("A9_Score", 0)),
            'A10_Score': int(request.form.get("A10_Score", 0)),
            'age': int(request.form.get("age", 0)),
            'gender': int(request.form.get("gender", 0)),
            'ethnicity': request.form.get("ethnicity", ""),
            'jaundice': int(request.form.get("jaundice", 0)),
            'austim': int(request.form.get("austim", 0)),
            'contry_of_res': request.form.get("contry_of_res", ""),
            'used_app_before': int(request.form.get("used_app_before", 0)),
            'result': int(request.form.get("result", 0)),
            'relation': request.form.get("relation", "")
        }

        input_df = pd.DataFrame([input_data])

        for col in ['ethnicity', 'contry_of_res', 'relation']:
            input_df[col] = input_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'Others')
            input_df[col] = label_encoders[col].transform(input_df[col])

        feature_order = [
            'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'austim',
            'contry_of_res', 'used_app_before', 'result', 'relation'
        ]
        input_df = input_df[feature_order]

        input_array = input_df.values.reshape(1, -1)

        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0]

        result = "Autistic" if prediction[0] == 1 else "Not Autistic"
        confidence = f"{max(probability) * 100:.1f}%"

        return render_template(
            "result.html",
            prediction=result,
            confidence=confidence,
            features=input_data,
            username=session['username']  # Pass username to template
        )

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)