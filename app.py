import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session, escape
import numpy as np
import pandas as pd
from joblib import load
app = Flask(__name__)

#Load the model
potato_model = pickle.load(open("potato_model.pkl", "rb"))
# transformer = pickle.load(open("potato_tranformer.pkl", "rb"))
columns = pickle.load(open("crop_columns.pkl", "rb"))
transformer = load(filename="potato_transformer.joblib")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    #converting to list
    data = np.array(list(data.values()))
    print(data)
    # reshaping 
    data = data.reshape(1, -1)
    df = pd.DataFrame(data, columns=columns)
    print(df)
    transformed_df = transformer.transform(df)
    #predicting the output
    output = potato_model.predict(transformed_df)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = [x for x in request.form.values()]
    print(data)
    data = np.array(data)
    #reshaping
    data = data.reshape(1, -1)
    #creating dataframe
    df = pd.DataFrame(data, columns=columns)
    print(df)
    transformed_df = transformer.transform(df)
    #predicting output
    output = potato_model.predict(transformed_df)[0]
    print(output)

    return render_template("home.html", prediction_text = f"Water Required : {output}")

if __name__ == "__main__":
    app.run(debug=True)