from flask import Flask,render_template,url_for,request, send_from_directory
from nlpModel import model
import os

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER



@app.route('/')
def home():
	img1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
	return render_template('dashboard.html', img1 = img1_path)

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		txt = request.form['comment']
		results = model.predict(txt)
	return render_template('result.html',results = results)



if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)