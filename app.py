import os
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("best_meal_budget_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dietary_pref = request.form['diet']
        eating_out = float(request.form['eating_out'])
        age = request.form['age']
        fashion_spend = float(request.form['fashion_spend'])

        input_data = pd.DataFrame({
            'Eating Out Per week': [eating_out],
            'Fashion spend per month(in-inr)\nProvide values in integer   between (0-2000 numeric)': [fashion_spend],
            'Dietary Preference': [dietary_pref],
            'Age': [age]
        })

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_encoded)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted Meal Budget: ₹{prediction}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
