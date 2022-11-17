# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
import json

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict_value():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json() # Get data posted as a json
        X = np.zeros(2)
        X[0] = data['year']
        X[1] = data['month']
        X = X.reshape(1,-1)
        print(X)
        #data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(X)  # runs globally loaded model on the data
    #return str(prediction[0])
    result_json = json.dumps({"prediction": prediction[0]})
    print(result_json)

    return (result_json)

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='localhost', port=5000)