import json, time
import pickle

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# model_name = "Energy Consumption Model"
# model_version = 1
#
# model = mlflow.pyfunc.load_model(
#     model_uri=f"models:/{model_name}/{model_version}"
# )
model = pickle.load(open("trained_model.pkl", "rb"))

with open("responses.json", mode='w', encoding='utf-8') as f:
    json.dump([], f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    results = [round(prediction[0][0], 2), round(prediction[0][1], 2)]
    # pd.DataFrame(results, columns=['results']).to_csv('results.csv')
    response = {
        "Datetime": time.time(),
        "Values": results,
        "Model Version": "model_version"
    }
    write_json("responses.json", response)

    return render_template('index.html', prediction_text=results)


def write_json(filename, data):
    with open(filename) as fp:
        listObj = json.load(fp)

    listObj.append(data)

    with open(filename, 'w') as json_file:
        json.dump(listObj, json_file, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080 , debug=True)
