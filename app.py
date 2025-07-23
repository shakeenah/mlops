
from flask import Flask, request,render_template
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        data={
            "City": request.form.get["City"],
            "RoomType": request.form['RoomType'],
            "Bedroom": float(request.form['Bedroom']),
            "Bathroom": float(request.form['Bathroom']),
            "GuestsCapacity": int(request.form['GuestsCapacity']),
            "HasWifi": int(request.form['HasWifi']),
            "HasAC": int(request.form['HasAC']),
            "DistanceFromCityCenter": float(request.form['DistanceFromCityCenter'])
        }
        df=pd.DataFrame([data])
        prediction=model.predict(df)
        return render_template('index.html', prediction_text=f"Predicted price: {round(prediction[0], 2)}")
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
