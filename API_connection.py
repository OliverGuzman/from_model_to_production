import tensorflow as tf
from flask import Flask, request, json
import numpy as np
import pandas as pd
from waitress import serve
import sys
import scipy.stats
from data_processing import preprocessing_input


#import model
model = tf.keras.models.load_model(f'.../Project_Model_Production/mlruns/{sys.argv[1]}/{sys.argv[2]}/artifacts/Anomaly_model/data/model.keras')

#create a flask instance
app = Flask(__name__)
variable_mean = float(list(open(f".../Project_Model_Production/mlruns/{sys.argv[1]}/{sys.argv[2]}/metrics/MAE Mean Score","rt").read().split(" "))[1])
variable_std = float(list(open(f".../Project_Model_Production/mlruns/{sys.argv[1]}/{sys.argv[2]}/metrics/Standard Deviation","rt").read().split(" "))[1])

x=variable_mean+variable_std
limit_score= scipy.stats.norm.cdf(x,variable_mean,variable_std)

#add API method to predict anomalies
@app.route("/anomaly_detection", methods=["POST"])
def anomaly_detection():
    #impot json and preprocess
    j = request.json
    df = pd.DataFrame.from_dict(j)
    df = preprocessing_input(df)
    model_input = df.to_numpy()

    #prediction and response
    model_output = model.predict(model_input)
    difference = tf.keras.losses.mae(model_input, model_output)
    reply_model = scipy.stats.norm.cdf(difference,variable_mean,variable_std)
    response = {'Percentage of being an anomaly':reply_model[0],'MAE':difference.numpy()[0]}
    
    return json.dumps(str(response))
#start the app
if __name__ == "__main__":
    serve(app,host="0.0.0.0", port=8000)