from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3
import joblib
import pandas as pd
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Load Trained Model
model = joblib.load('model.pkl')

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age REAL,
            bmi REAL,
            hba1c REAL,
            glucose REAL,
            risk_probability REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Get KPIs
    cursor.execute('SELECT COUNT(*), AVG(risk_probability) FROM screenings')
    total, avg_risk = cursor.fetchone()
    
    # Get recent screening history
    cursor.execute('SELECT id, timestamp, age, bmi, hba1c, glucose, risk_probability FROM screenings ORDER BY id DESC LIMIT 10')
    history = cursor.fetchall()
    conn.close()
    
    total = total if total else 0
    avg_risk = round(avg_risk, 1) if avg_risk else 0.0
    
    return render_template('index.html', total_screened=total, avg_risk=avg_risk, history=history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    # Gather form inputs
    gender = request.form['gender']
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    hba1c = float(request.form['hba1c'])
    glucose = float(request.form['glucose'])
    smoking = request.form['smoking']
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])

    # Construct input DataFrame matching your training features
    input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose]], 
                              columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    
    # Predict Probability
    prob = model.predict_proba(input_data)[0][1] * 100
    risk_score = round(prob, 2)

    # Save record to SQLite Database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO screenings (age, bmi, hba1c, glucose, risk_probability) VALUES (?, ?, ?, ?, ?)',
                   (age, bmi, hba1c, glucose, risk_score))
    conn.commit()
    conn.close()

    # Generate Gemini AI Clinical Insights
    prompt = f"""
    Act as a clinical triage assistant. A patient has been screened for diabetes risk.
    Clinical Metrics:
    - Age: {age}
    - BMI: {bmi}
    - HbA1c Level: {hba1c}%
    - Blood Glucose Level: {glucose} mg/dL
    - Calculated Diabetes Risk Probability: {risk_score}%
    
    Provide a concise, non-diagnostic triage summary, explain what HbA1c and glucose mean in this context, and list 2-3 questions they can ask their doctor. Format with markdown headings.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt
        )
        ai_summary = response.text
    except Exception as e:
        ai_summary = "AI insights temporarily unavailable."

    return render_template('result.html', risk_score=risk_score, ai_summary=ai_summary, age=age, bmi=bmi, hba1c=hba1c, glucose=glucose)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    context = data.get('context', '')

    chat_prompt = f"""
    You are a healthcare assistant chatbot. You previously provided this context to a patient:
    {context}

    The patient is asking a follow-up question:
    "{user_message}"

    Remember: Never deliver a medical diagnosis or tell them what medical action to take. Act like a nurse explaining terms or suggesting questions for a doctor. Keep it concise.
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=chat_prompt
        )
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'reply': "Chat response temporarily unavailable."})

if __name__ == '__main__':
    app.run(debug=True)