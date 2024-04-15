import tensorflow as tf
import numpy as np
import mlflow
from evaluation import prediction
from sklearn.metrics import accuracy_score
from data_processing import data_preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

def train_model(df, model_setting):
    #Input data and data split
    x_good, x_bad = data_preprocessing(df)
    not_anomalies = df.loc[df["anomaly"]==0]["anomaly"].to_numpy()

    x_good_train, x_good_test = train_test_split(x_good, test_size=0.25, random_state=42)
    x_good_train, x_good_validation = train_test_split(x_good_train, random_state=42)

    mlflow.set_experiment("anomaly_detection")
    experiment = mlflow.get_experiment_by_name("anomaly_detection")
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id = run.info.run_id):
        #neural network settings
        model = Sequential()
        model.add(Dense(model_setting.get("first_layer"), activation='relu'))
        model.add(Dense(model_setting.get("hidden_layer"), activation='relu'))
        model.add(Dense(model_setting.get("second_layer"), activation='relu'))
        model.add(Dense(model_setting.get("output_layer"))) 
        model.compile(loss=model_setting.get("loss"), optimizer=model_setting.get("optimizer"))
        #train model
        model.fit(x_good_train,x_good_train,epochs=model_setting.get("epochs"), validation_data=(x_good_validation,x_good_validation))

        preds = model.predict(x_good_test)
        mae_values = tf.keras.losses.mae(preds, x_good_test)
        mae_score = np.mean(mae_values)

        threshold = mae_score + np.std(mae_values)

        preds_final = prediction(model,x_good,threshold)
        accuracy = accuracy_score(not_anomalies,preds_final)

        mlflow.log_metric("MAE Mean Score", mae_score)
        mlflow.log_metric("Standard Deviation", np.std(mae_values))
        mlflow.log_metric("Threshold", threshold)
        mlflow.log_metric("Accuracy Score", accuracy)
        mlflow.log_params(model_setting)
        signature = mlflow.models.infer_signature(x_good,x_good)
        mlflow.tensorflow.log_model(model,"Anomaly_model", signature=signature,registered_model_name="AnomalyDectitionModel")
    
    return mae_score, accuracy









