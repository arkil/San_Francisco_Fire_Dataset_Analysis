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
from PreprocessingStep import preprocessInput	
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
		#img = img[:,:,:3]
		#img = img.reshape(1, -1)
		file_inp = preprocessInput(read_df)
		
		print(file_inp.shape)

		# make prediction on new image
		prediction = model.predict(file_inp)
	
		# squeeze value from 1D array and convert to string for clean return
		label = str(np.squeeze(prediction))

		# switch for case where label=10 and number=0
		# if label=='10': label='0'

		return render_template('index.html', label=label)
   
if __name__ == '__main__':
    model = joblib.load('sf.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)