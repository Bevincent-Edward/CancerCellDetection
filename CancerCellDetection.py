from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('CancerCellDetection.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()
        data = {key: [float(value)] for key, value in data.items()}

        # Convert form data to DataFrame
        input_data = pd.DataFrame.from_dict(data)

        # Make prediction
        prediction = model.predict(input_data)

        # Determine result
        result = "Not Cancerous" if prediction[0] == 2 else "Cancerous"

        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)