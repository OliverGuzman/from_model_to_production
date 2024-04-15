import mlflow
import pandas as pd
from data_processing import extract_sensor_info, load_raw_data
from model_training import train_model

def pipeline():
    mlflow.set_experiment("anomaly_detection")
    path_file = pd.read_csv(".../Project_Model_Production/data/export-2024-04-0216_1716_17_28.csv")
    raw_data_file = load_raw_data(path_file)
    dataframe = extract_sensor_info(raw_data_file)

    model_setting = {
        "first_layer":10,
        "hidden_layer":4,
        "second_layer":10,
        "output_layer":5,
        "loss":"mae",
        "optimizer":"adam",
        "epochs":50
    }

    mae_score, accuracy = train_model(dataframe, model_setting)
    
    print(f"Final model is trained. \nMAE Score: {mae_score}\nAccuracy: {accuracy}")

if __name__ == "__main__":
    pipeline()