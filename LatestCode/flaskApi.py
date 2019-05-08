# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:10:26 2018

@author: arkil
"""

import flask
from flask import Flask, render_template, request
from sklearn.externals import joblib
from scipy import misc
import numpy as np
import pandas as pd
from SFFirePredLibraries import preprocessInput
from SFFirePredLibraries import getTrainColumns
from SFFirePredLibraries import imputeColumnValues	
app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

   
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file: return render_template('index.html', label="No file")
        read_df = pd.read_csv(file, sep='\t')
        file_inp = preprocessInput(read_df)
        train_cols = getTrainColumns()
        mod_test_df = imputeColumnValues(train_cols, file_inp)
        print(file_inp.shape)
        modelused = request.form["modelused"]
        if modelused == "lasso":
            prediction = lasso_model.predict(mod_test_df)
        elif modelused == "randomforest":
            prediction = rf_model.predict(mod_test_df)
        elif modelused == "ridge":
            prediction = ridge_reg_model.predict(mod_test_df)
        else:
            prediction = gradient_boosting_model.predict(mod_test_df)
		#prediction = model.predict(mod_test_df)
        label = str(np.squeeze(prediction))
        label = int(label)
        return render_template('index.html', label=label)
   
if __name__ == '__main__':
    lasso_model= joblib.load('sf_reg_lasso.pkl')
    rf_model = joblib.load('rforest.pkl')
    ridge_reg_model = joblib.load('sf_reg_ridge.pkl')
    gradient_boosting_model = joblib.load('sf_reg_gbr.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)