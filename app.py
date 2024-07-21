import logging
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# إعداد سجل التصحيح
logging.basicConfig(level=logging.DEBUG)

# تحميل النموذج
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")
        
        features = pd.DataFrame([data])
        logging.debug(f"DataFrame: {features}")
        
        prediction = model.predict(features)
        logging.debug(f"Prediction: {prediction}")
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
