from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('iris_model.pkl')
feature_cols = joblib.load('iris_model_cols.pkl')

# Define the mapping of predicted numerical values to species names
species_mapping = {0: 'Iris-setosa', 1: 'Iris-Versicolor', 2: 'Iris-virginica'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input from the request
        input_data = request.get_json()

        # Validate if the required fields are present
        if not all(feature in input_data for feature in feature_cols):
            return jsonify({'error': 'Missing features'}), 400

        # Prepare the input data as a NumPy array
        input_array = np.array([[input_data[feature] for feature in feature_cols]])

        # Make a prediction
        predicted_species_num = model.predict(input_array)[0]

        # Map the numerical prediction to species name
        predicted_species = species_mapping.get(predicted_species_num, 'Unknown Species')

        # Return the result as JSON
        result = {'prediction': predicted_species}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
