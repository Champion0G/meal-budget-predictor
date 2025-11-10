from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and the columns it was trained with
model = joblib.load("best_meal_budget_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        dietary_pref = request.form['diet']
        eating_out = float(request.form['eating_out'])
        age = request.form['age']
        fashion_spend = float(request.form['fashion_spend'])

        # Build the input data
        input_data = pd.DataFrame({
            'Eating Out Per week': [eating_out],
            'Fashion spend per month(in-inr)\nProvide values in integer   between (0-2000 numeric)': [fashion_spend],
            'Dietary Preference': [dietary_pref],
            'Age': [age]
        })

        # One-hot encode like training
        input_encoded = pd.get_dummies(input_data)

        # Align columns to training columns
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted Meal Budget: ₹{prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
