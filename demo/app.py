import os
import sys

sys.path.append('../models')
import predict as pr

from flask import Flask, request, redirect, render_template


app = Flask(__name__)

path = '.\static'

predictor = pr.Predictor()

@app.get('/')
def home():
	return render_template('template.html')

@app.post('/')
def get_file():
	file_names = os.listdir('./static')
	for f in file_names:
		os.remove(f'./static/{f}')

	if 'img' not in request.files:
		return redirect('/')
    
	file = request.files['img']

	if file and ('jpg' in file.filename):
		file.save(os.path.join(path, file.filename))
		img = os.path.join(path, file.filename)
	
	encoded = predictor.encode(img)
	preds, preds_beam = predictor.decode(encoded)

	return render_template('template.html', 
							img=img, 
							preds=preds, 
							preds_beam=preds_beam)