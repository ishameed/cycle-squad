from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)
model = load('random_forest_member_type_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features_list = data['features']
    features_df = pd.DataFrame(features_list, columns=['rideable_type', 'day_of_week', 'month', 'season', 'trip_duration_mins', 'hour', 'trips', 'day_of_month'])
    predictions = model.predict(features_df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=5000)
