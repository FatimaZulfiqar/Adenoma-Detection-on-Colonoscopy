from flask import Flask,request, url_for, redirect, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# set path to model files here
model1 = pickle.load(open('model/knn_model.pkl', 'rb'))
model2 = pickle.load(open('model/svr_model.pkl', 'rb'))

cols = ['age', 'Hepatitis C', 'colonoscopy', 'No of polyps', 'aspirin',
       'smoking', 'alcohol', 'ethnicity', 'bmi', 'gender']

#category_col =['HCV Genotype', 'gene polymorphism']

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    result1 = model1.predict(to_predict)
    result2 = model2.predict(to_predict)
    return result1[0], result2[0]

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        pred1, pred2 = ValuePredictor(to_predict_list)
        pred2 = np.round(pred2,2)
        print("Colonoscopy = ",pred1)
        print("\nno of polyp = ",pred2)
        if int(pred1) == 0:
            prediction = "0 - No adenoma polyp"
        elif int(pred1) == 1:
            prediction = "1 - yes polyp"

        return render_template('home.html',pred="Colonoscopy: {},"
                                                " No of polyps = {}".format(prediction,pred2))

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction1 =model1.predict(data_unseen)
    prediction2 = model2.predict(data_unseen)
    output1 = prediction1
    output2 = np.round(prediction2,2)
    return jsonify(output1,output2)
if __name__ == '__main__':
    app.run(debug=True)
