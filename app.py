from flask import Flask, request, jsonify
import requests
from joblib import load
import pandas as pd
import os
import boto3
from werkzeug.exceptions import InternalServerError

app = Flask(__name__)
# model = load('random_forest_member_type_model_final.joblib')

# # AWS S3 Configuration
# BUCKET_NAME = 'heroku-model'  # replace with your bucket name
# MODEL_FILE_NAME = 'random_forest_member_type_model_final.joblib'  # replace with your model file name
# MODEL_PATH = '/tmp/' + MODEL_FILE_NAME  # '/tmp' is writable on Heroku

MODEL_URL = 'https://heroku-model.s3.amazonaws.com/random_forest_member_type_model.joblib' 
MODEL_PATH = 'random_forest_member_type_model_final.joblib'

if not os.path.isfile(MODEL_PATH):
    print("Downloading model from S3...")
    response = requests.get(MODEL_URL, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        model = load('random_forest_member_type_model_final.joblib')
    else:
        print(f"Failed to download model, status code: {response.status_code}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features_list = data['features']
    features_df = pd.DataFrame(features_list, columns=['rideable_type', 'day_of_week', 'month', 'season', 'trip_duration_mins', 'hour', 'trips', 'day_of_month'])
    predictions = model.predict(features_df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)