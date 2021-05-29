from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import json
from nsetools import Nse
import datetime
from nsepy import get_history
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

nse = Nse()
app = Flask(__name__,  static_folder="build/static", template_folder="build")
cors = CORS(app)

@app.route("/")
def frontend():
    return render_template('index.html')


@app.route('/api/predict/<symbol>')
def get_prediction(symbol):
    data = request.get_json()
    
    today = datetime.date.today()
    data = get_history(symbol="RELIANCE", start = today - datetime.timedelta(days = 15), end = today)

    last_week = data[(today - datetime.timedelta(days = 15)):(today - datetime.timedelta(days = 7))]
    this_week = data[(today - datetime.timedelta(days = 7)):today]

        #Let's select our features
    features = ['High','Turnover','VWAP', 'Volume']
    predictor = 'Close'
    X = last_week.loc[:,features]
    y = last_week.loc[:,predictor]

    new_X = this_week.loc[:,features]
    new_y = this_week.loc[:,predictor]

    X_train, _, y_train, _ = train_test_split(X, y, test_size = 0.1, random_state = 0)

    Classifier = DecisionTreeRegressor()

    Classifier.fit(X_train, y_train)

    pred_y = Classifier.predict(new_X)

    result = {
        "dates": list(this_week.reset_index()["Date"].apply(lambda x: x.strftime("%d/%b"))),
        "truth": list(new_y),
        "predictions":list(pred_y)
    }
    return jsonify(result)


@app.route('/api/gainers')
def get_gainers():
    top_gainers = nse.get_top_gainers()
    gainer_names = list()
    for gainer in top_gainers:
        gainer_names.append(gainer["symbol"])

    print(gainer_names) 
    return json.dumps(gainer_names)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)#port=443, ssl_context='adhoc')