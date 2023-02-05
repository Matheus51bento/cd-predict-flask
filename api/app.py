import joblib
from flask import Flask, jsonify, request
from tensorflow import keras
import functools
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
model = keras.models.load_model("../cdipredic/model/model.h5")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)


@app.route("/api", methods=["POST"])
def login():
    # dinheiro = request.args.get('dinheiro')
    vetory = [1.44, 1.44, 1.44, 1.28, 1.49, 1.39, 1.3, 1.4, 1.22, 1.28, 1.22, 1.19, 1.26, 1.01, 1.25, 1.18, 1.33, 1.27, 1.5, 1.6, 1.32, 1.53, 1.39, 1.39, 1.53, 1.25, 1.37, 1.48, 1.4, 1.31, 1.53, 1.45, 1.38, 1.64, 1.53, 1.73, 1.97, 1.83, 1.77, 1.87, 1.96, 1.85, 2.08, 1.76, 1.67, 1.63, 1.34, 1.37, 1.26, 1.08, 1.37, 1.17, 1.22, 1.22, 1.28, 1.29, 1.24, 1.21, 1.25, 1.48, 1.38, 1.22, 1.52, 1.41, 1.5, 1.58, 1.51, 1.65, 1.5, 1.4, 1.38, 1.47, 1.43, 1.14, 1.42, 1.08, 1.28, 1.18, 1.17, 1.25, 1.05, 1.09, 1.02, 0.98, 1.08, 0.87, 1.05, 0.94, 1.02, 0.9, 0.97, 0.99, 0.8, 0.92, 0.84, 0.84, 0.92, 0.8, 0.84, 0.9, 0.87, 0.95, 1.06, 1.01, 1.1, 1.17, 1.0, 1.11, 1.04, 0.85, 0.97, 0.84, 0.77, 0.75, 0.78, 0.69, 0.69, 0.69, 0.66, 0.72, 0.66, 0.59, 0.76, 0.66, 0.75, 0.79, 0.86, 0.89, 0.84, 0.81, 0.81, 0.93, 0.86, 0.84, 0.92, 0.84, 0.99, 0.95, 0.97, 1.07, 0.94, 0.88, 0.86, 0.9, 0.89, 0.74, 0.81, 0.7, 0.73, 0.64, 0.68, 0.69, 0.54, 0.61, 0.54, 0.53, 0.59, 0.48, 0.54, 0.6, 0.58, 0.59, 0.71, 0.7, 0.7, 0.8, 0.71, 0.78, 0.84, 0.78, 0.76, 0.82, 0.86, 0.82, 0.94, 0.86, 0.9, 0.94, 0.84, 0.96, 0.93, 0.82, 1.04, 0.95, 0.98, 1.07, 1.18, 1.11, 1.11, 1.11, 1.06, 1.16, 1.05, 1.0, 1.16, 1.05, 1.11, 1.16, 1.11, 1.21, 1.11, 1.05, 1.04, 1.12, 1.08, 0.86, 1.05, 0.79, 0.93, 0.81, 0.8, 0.8, 0.64, 0.64, 0.57, 0.54, 0.58, 0.46, 0.53, 0.52, 0.52, 0.52, 0.54, 0.57, 0.47, 0.54, 0.49, 0.49, 0.54, 0.49, 0.47, 0.52, 0.54, 0.47, 0.57, 0.5, 0.46, 0.48, 0.38, 0.37, 0.38, 0.29, 0.34, 0.28, 0.24, 0.21, 0.19, 0.16, 0.16, 0.16, 0.15, 0.16, 0.15, 0.13, 0.2, 0.21, 0.27, 0.31, 0.36, 0.43, 0.44, 0.49, 0.59, 0.77, 0.73, 0.76, 0.93, 0.83, 1.03, 1.02, 1.03, 1.17, 1.07, 1.02, 1.02, 1.12, 1.12, 0.1]
    vetory = moving_average(vetory,12)

    new_vetor = []
    count = 0
    while len(vetory) > count:
        new_vetor.append([vetory[count]])
        count += 1

    dataset = new_vetor

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.70)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    look_back = 10
    testX, testY = create_dataset(test, look_back)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    dinheiro = int(request.form.get('dinheiro'))
    dinheiro = dinheiro * -1

    predictNextNumber = model.predict(testX[dinheiro:], verbose=1)

    return jsonify({"cdi": str(list(predictNextNumber.flatten()))})

if __name__ == "__main__":
  app.run(debug=True)