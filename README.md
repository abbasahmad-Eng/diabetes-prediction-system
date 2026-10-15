# Diabetes Prediction & Patient Assistance System

An end-to-end web application that screens patients for diabetes risk using Machine Learning and provides AI-driven clinical triage explanations powered by Google Gemini.

## Features
- **Diabetes Risk Prediction**: Uses a trained Gradient Boosting classifier to evaluate clinical biomarkers (Age, BMI, HbA1c Level, Blood Glucose Level, Hypertension, Heart Disease, etc.) and predict diabetes risk probability.
- **Generative AI Clinical Insights**: Integrates the Google Gemini API to generate structured, non-diagnostic triage summaries, term explanations, and suggested doctor questions.
- **Persistent Database**: Automatically logs all diabetes risk screenings into a local SQLite database (`database.db`).
- **Interactive Dashboard**: Displays aggregate screening analytics and historical prediction records.

## Project Structure
```text
├── templates/
│   ├── index.html                  # Main dashboard & diabetes input form
│   └── result.html                 # Prediction results & AI clinical summary
├── app.py                          # Flask backend & Gemini API integration
├── train_model.py                  # ML model training script
├── EDA_and_Model_Comparison.ipynb  # Exploratory Data Analysis & model comparison notebook
├── diabetes_prediction_dataset.csv # Diabetes dataset
├── model.pkl                       # Trained Scikit-Learn pipeline
└── database.db                     # Local SQLite database




## Dataset Source & Licence
* **Dataset Name**: Diabetes Prediction Dataset
* **Source**: Kaggle (Diabetes Prediction Dataset repository)
* **Licence**: Open-source / Creative Commons license for educational and research use.

---

## Capstone Write-Up & Answers

### 1. Which ML problem did you pick within this project, and why that one over the others on the list?
We selected **Diabetes Prediction**. We chose this over other healthcare problems because clean, real-world clinical biomarker data (such as HbA1c levels, blood glucose, and BMI) is widely accessible, and early identification of at-risk patients allows case managers to intervene before costly complications or readmissions occur.

### 2. Where did your dataset come from, and what did the EDA tell you before you built anything?
The dataset originated from Kaggle (Diabetes Prediction Dataset, available under an open-source license). Before building the model, our Exploratory Data Analysis (EDA) evaluated missing values, data types, and feature correlations. We found that features like `HbA1c_level` and `blood_glucose_level` showed the strongest positive correlation with diabetes risk. We also analyzed the class distribution to ensure our training pipeline could properly differentiate between high-risk and low-risk patient profiles.

### 3. Which algorithm(s) did you try, which one did you ship, and what evaluation metrics justify that choice?
We benchmarked three classification algorithms during development: Logistic Regression, Random Forest, and Gradient Boosting. We shipped **Gradient Boosting** because it achieved the highest overall performance, yielding an accuracy of $97.24\%$ and an exceptional ROC-AUC score of $0.9791$, outperforming the other models in capturing complex non-linear relationships among clinical biomarkers.

### 4. How exactly does the Gen AI layer use the model's output? Show a real example.
The Google Gemini API (`gemini-3.8-flash`) is dynamically triggered whenever a prediction is made. It takes the specific patient inputs and the model's calculated risk score to generate a tailored, non-diagnostic triage summary, lab metric explanations, and suggested doctor questions.
* **Real Example from our App**:
  * **Patient Input**: Age: 45, BMI: 23.0, HbA1c: 5.6%, Blood Glucose: 180.0 mg/dL.
  * **Model Prediction**: Calculated Diabetes Risk Probability: **3.21%** (Low Risk).
  * **Generated AI Nurse Summary**: *"Based on your screening metrics, your calculated statistical risk for diabetes is low (3.21%), and your BMI of 23.0 falls within the healthy range. However, your point-in-time blood glucose reading was elevated at 180.0 mg/dL... Your doctor will need to evaluate this in the context of when you last ate..."*