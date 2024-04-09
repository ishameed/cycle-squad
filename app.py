from flask import Flask, request, jsonify
import requests
from joblib import load
import pandas as pd
import os
import boto3
from werkzeug.exceptions import InternalServerError
from threading import Lock

# app = Flask(__name__)
model = None
model_lock = Lock()


MODEL_URL = 'https://heroku-model.s3.amazonaws.com/random_forest_member_type_model.joblib' 
MODEL_PATH = 'random_forest_member_type_model_final.joblib'

# if not os.path.isfile(MODEL_PATH):
#     print("Downloading model from S3...")
#     response = requests.get(MODEL_URL, stream=True)

#     # Check if the request was successful
#     if response.status_code == 200:
#         with open(MODEL_PATH, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         model = load('random_forest_member_type_model_final.joblib')
#     else:
#         print(f"Failed to download model, status code: {response.status_code}")



def load_model():
    global model
    with model_lock:  # Use a lock to ensure only one worker does this
        if model is None:
            if not os.path.exists(MODEL_PATH):
                print("Downloading model from S3...")
                response = requests.get(MODEL_URL, stream=True)

                if response.status_code == 200:
                    with open(MODEL_PATH, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Failed to download model, status code: {response.status_code}")
            print("Loading model...")
            model = load(MODEL_PATH)

def create_app():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        if model is None:
            load_model() 
        data = request.get_json()
        features_list = data['features']
        features_df = pd.DataFrame(features_list, columns=['rideable_type', 'day_of_week', 'month', 'season', 'trip_duration_mins', 'hour', 'trips', 'day_of_month'])
        predictions = model.predict(features_df)
        return jsonify(predictions.tolist())

app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)