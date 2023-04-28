import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session, escape
import numpy as np
import pandas as pd
from joblib import load
app = Flask(__name__)

#Load the model
potato_model = pickle.load(open("potato_model.pkl", "rb"))
wheat_model = pickle.load(open("wheat_model.pkl", "rb"))
# transformer = pickle.load(open("potato_tranformer.pkl", "rb"))
columns = pickle.load(open("crop_columns.pkl", "rb"))
potato_transformer = load(filename="potato_transformer.joblib")
wheat_transformer = load(open("wheat_transformer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    #converting to list
    data = list(data.values())
    crop_type = data[0]
    data = data[1:]
    data = np.array(data)
    print(data)
    # reshaping 
    data = data.reshape(1, -1)
    df = pd.DataFrame(data, columns=columns)
    print(df)

    crop_type = crop_type.upper()
    output = 0
    match crop_type:
        case "POTATO":
            transformed_potato = potato_transformer.transform(df)
            output = potato_model.predict(transformed_potato)[0]
        case "WHEAT":
            transformed_wheat = wheat_transformer.transform(df)
            output = wheat_model.predict(transformed_wheat)[0]
        case _ :
            print("Not valid crop")

    # transformed_df = potato_transformer.transform(df)
    # #predicting the output
    # output = potato_model.predict(transformed_df)

    return jsonify(output)

@app.route("/predict", methods=["POST"])
def predict():
    data = [x for x in request.form.values()]
    print(data)
    crop_type = data[0]
    data = data[1:]
    data = np.array(data)
    #reshaping
    data = data.reshape(1, -1)
    #creating dataframe
    df = pd.DataFrame(data, columns=columns)
    #selecting crop type
    crop_type = crop_type.upper()

    output = 0
    match crop_type:
        case "POTATO":
            transformed_potato = potato_transformer.transform(df)
            output = potato_model.predict(transformed_potato)[0]
        case "WHEAT":
            transformed_wheat = wheat_transformer.transform(df)
            output = wheat_model.predict(transformed_wheat)[0]
        case _ : 
            print("Not Valid Crop")
        
    # transformed_df = potato_transformer.transform(df)
    # #predicting output
    # output = potato_model.predict(transformed_df)[0]
    print(output)

    return render_template("home.html", prediction_text = f"Water Required : {output}")

if __name__ == "__main__":
    app.run(debug=True)