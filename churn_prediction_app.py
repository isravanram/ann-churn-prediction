# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
import datetime
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

def add_break_line():
  print("\n------------------------------------------------\n")


def save_pickle_file(file_name: str,output): 
  with open(file_name,'wb') as file:
    pickle.dump(output,file)

def load_pickle_file(pickle_file_name):
  with open(pickle_file_name,'rb') as file:
    return pickle.load(file)

def build_churn_prediction_model():
  data = pd.read_csv("Churn_Modelling.csv")
  data.drop(["RowNumber","CustomerId","Surname"],axis=1,inplace=True)

  # Encoding Gender column values to 0,1
  label_encoder = LabelEncoder()
  data['Gender'] = label_encoder.fit_transform(data["Gender"])
  print(f"Encoded gender: \n {data['Gender']}")
  add_break_line()
  one_hot_encoder = OneHotEncoder()
  # geo_encoded = one_hot_encoder.fit_transform(data['Geography']) # This gives error since fit_transform method expects a pandas dataframe as the input
  geo_encoded = one_hot_encoder.fit_transform(data[["Geography"]])
  print(f"Encoded geography: \n {geo_encoded}")
  geo_features = one_hot_encoder.get_feature_names_out(['Geography'])
  geo_encoded_df = pd.DataFrame(geo_encoded.toarray(),columns=geo_features)
  print(f"Geo encoded df: \n {geo_encoded_df}")

  # Concatenating the two dataframes
  data = pd.concat([data.drop(['Geography'],axis=1),geo_encoded_df],axis=1)

  save_pickle_file('label_encoder.pkl',label_encoder)
  save_pickle_file('one_hot_encoder.pkl',one_hot_encoder)

  # Dependent and independent features
  x= data.drop(['Exited'],axis=1)
  y = data['Exited']

  from sklearn.model_selection import train_test_split
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

  standard_scaler = StandardScaler()
  standard_scaler.fit_transform(x_train)
  standard_scaler.transform(x_test)
  save_pickle_file('standard_scaler.pkl',standard_scaler)

  model = Sequential([
      Dense(64,activation='relu',input_shape=(x_train.shape[1],)),
      Dense(64,activation='relu'),
      Dense(1,activation='sigmoid')   
  ])

  print(model.summary())

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  loss = tf.keras.losses.BinaryCrossentropy()

  model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
  logs_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  print(x_train.shape)
  print(x_test.shape)
  print(y_train.shape)
  print(y_test.shape)

  tensorboard = TensorBoard(log_dir = logs_dir,histogram_freq=1)
  early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

  history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,
                      callbacks=[tensorboard,early_stopping_callback])

  model.save('churn_prediction_model.h5')

def import_parameters():
  label_encoder = load_pickle_file("label_encoder.pkl")
  one_hot_encoder = load_pickle_file("one_hot_encoder.pkl")
  standard_scaler = load_pickle_file("standard_scaler.pkl")
  model = load_model('churn_prediction_model.h5')
  return label_encoder,one_hot_encoder,standard_scaler,model

def predict_churn_probability(input_data,parameters):
  label_encoder,one_hot_encoder,standard_scaler,model = parameters[0],parameters[1],parameters[2],parameters[3]
  geo_encoded = one_hot_encoder.transform([[input_data['Geography']]]).toarray()
  geo_encoded_df = pd.DataFrame(geo_encoded,columns = one_hot_encoder.get_feature_names_out(['Geography']))
  print(geo_encoded_df)
  input_df = pd.DataFrame([input_data])
  input_df = pd.concat([input_df.drop('Geography',axis=1),geo_encoded_df],axis=1)
  # input_df = pd.concat([input_df.drop('Geography',axis=1),geo_encoded_df])

  print(input_df)
  input_df['Gender'] = label_encoder.transform(input_df['Gender'])
  add_break_line()

  # Scaling
  input_df = standard_scaler.transform(input_df)
  print(input_df)

  prediction=model.predict(input_df)
  predict_proba = prediction[0][0]
  if predict_proba > 0.5:
    return "Customer likely to churn",predict_proba
  else:
    return "Customer not likely to churn",predict_proba

if __name__ == '__main__':
  build_churn_prediction_model()
  label_encoder,one_hot_encoder,standard_scaler,model = import_parameters()
  parameters = [label_encoder,one_hot_encoder,standard_scaler,model]
  # Prediction
  input_data ={
      'CreditScore':600,
      'Geography':'France',
      'Gender':'Male',
      'Age':40,
      'Tenure':3,
      'Balance':60000,
      'NumOfProducts':2,
      'HasCrCard':1,
      'IsActiveMember':1,
      'EstimatedSalary':50000
  }
  print(predict_churn_probability(input_data,parameters))
    