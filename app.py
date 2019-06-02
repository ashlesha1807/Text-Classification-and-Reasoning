from flask import Flask,render_template,url_for,request
from nlpModel import model


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		txt = request.form['comment']
		results = model.predict(txt)
	return render_template('result.html',results = results)



if __name__ == '__main__':
	app.run(debug=True)