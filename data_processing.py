#%%
import pandas as pd
import tensorflow as tf

def load_raw_data(df):
    #import file and drop unneccesary columns
    df.drop(['location', 'metadata','time','group'], axis=1, inplace=True)
    df.dropna(inplace=True)

    #drops system message, work_off, work_on
    df=df.drop(list(df[df['variable'] == 'message_system'].index))
    df=df.drop(list(df[df['variable'] == 'work_off'].index))
    df=df.drop(list(df[df['variable'] == 'work_on'].index))

    #Converts value to float and id to strings
    df['value'] = df['value'].astype('float')
    df['id'] = df['id'].astype('string')
    df['variable'] = df['variable'].astype('string')

    return df

def extract_sensor_info(df):
    #creating dataframe for training
    df_training = pd.DataFrame(columns=['id','temperature_ini', 'temperature_mid','temperature_end','pressure','speed'])

    #filling training dataframe
    def fill_training_df(id_values):
        
        if id_values[:-1] in df_training.values:
            df_training.loc[df_training.loc[df_training["id"] == id_values[:-1]].index[0], df["variable"].loc[df["id"]==id_values].values[0]] = df["value"].loc[df["id"] == id_values].values[0]
            
        else:
            df2 = {'id': id_values[:-1], df["variable"].loc[df["id"]==id_values].values[0]: df["value"].loc[df["id"] == id_values].values[0]} 
            df_training.loc[len(df_training.index)] = df2

    #apply the filling of the training dataframe
    df["id"].apply(fill_training_df)

    #drop any row with "NaN"
    df_training.dropna(inplace=True)
    
    #filling anomaly field
    def fill_anomaly_df(id_values):
        
        if (df_training["speed"].loc[df_training["id"]==id_values].values[0] > 30.0 and
        df_training["pressure"].loc[df_training["id"]==id_values].values[0] <= 4000.0 and
        df_training["temperature_ini"].loc[df_training["id"]==id_values].values[0] <= 200.0 and
        df_training["temperature_mid"].loc[df_training["id"]==id_values].values[0] <= 200.0 and
        df_training["temperature_end"].loc[df_training["id"]==id_values].values[0] <= 200.0):
            return 0
        
        else:
            return 1

    df_training["anomaly"] = df_training["id"].apply(fill_anomaly_df) 

    return df_training

#Nornalization and separation of data
def data_preprocessing(df_training):
    #create an array with non-anomaly pieces
    df_good = df_training.loc[df_training["anomaly"]==0].copy()
    df_good.drop(["id","anomaly"],axis=1, inplace=True)
    x_good = df_good.values

    #min_val_good = tf.reduce_min(x_good)
    #max_val_good = tf.reduce_max(x_good)
    #x_good = (x_good - min_val_good) / (max_val_good - min_val_good)
    #x_good = x_good.numpy()

    #create an array with anomaly pieces
    df_bad = df_training.loc[df_training["anomaly"]==1].copy()
    df_bad.drop(["id","anomaly"],axis=1, inplace=True)
    x_bad = df_bad.values

    #min_val_bad = tf.reduce_min(x_bad)
    #max_val_bad = tf.reduce_max(x_bad)
    #x_bad = (x_bad - min_val_bad) / (max_val_bad - min_val_bad)
    #x_bad = x_bad.numpy()

    return x_good, x_bad

def preprocessing_input(df):
    df_input = pd.DataFrame(columns=['temperature_ini', 'temperature_mid','temperature_end','pressure','speed'])
    
    df2 = {'temperature_ini': df["value"].loc[df["variable"]=='temperature_ini'].values[0], 
           'temperature_mid': df["value"].loc[df["variable"]=='temperature_mid'].values[0],
           'temperature_end': df["value"].loc[df["variable"]=='temperature_end'].values[0],
           'pressure': df["value"].loc[df["variable"]=='pressure'].values[0],
           'speed': df["value"].loc[df["variable"]=='speed'].values[0]}
    df_input.loc[len(df_input.index)] = df2
    
    return df_input