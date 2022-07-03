import json, time
import mlflow
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

model_name = "Energy Model"
model_version = 10

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


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
        "Model Version": model_version
    }
    json_object = json.dumps(response, indent=4)

    with open("responses.json", "w") as outfile:
        outfile.write(json_object)

    return render_template('index.html', prediction_text=results)


if __name__ == "__main__":
    app.run(debug=True)
