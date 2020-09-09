import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("UberRide.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    myDict = request.form
    Priceperweek = int(myDict['Priceperweek'])
    Population = int(myDict['Population'])
    Monthlyincome = int(myDict['Monthlyincome'])
    Averageparkingpermonth = int(myDict['Averageparkingpermonth'])
    input_feature = [Priceperweek, Population, Monthlyincome, Averageparkingpermonth]
    prob = model.predict([input_feature])[0]
    return render_template('index.html', inf="Number of weekly rides should be {}".format(round(prob)))


if __name__ == "__main__":
    app.run(debug=True)


'''
This is another approach
if request.method == "POST":
        myDict = request.form
        Priceperweek = int(myDict['Priceperweek'])
        Population = int(myDict['Population'])
        Monthlyincome = int(myDict['Monthlyincome'])
        Averageparkingpermonth = int(myDict['Averageparkingpermonth'])
        input_feature = [Priceperweek, Population, Monthlyincome, Averageparkingpermonth]
        prob = model.predict([input_feature])[0]
        return render_template('show.html', inf=round(prob))
'''