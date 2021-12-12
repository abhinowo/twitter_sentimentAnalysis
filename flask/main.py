from flask import Flask, request
from flask_cors import CORS
from saved_model import predict
import json

app = Flask(__name__)
CORS(app, resources=r'/*')
@app.route('/')
def index():
    return "Sentiment analysis server is running"

@app.route('/predict', methods=['GET','POST'])
def predict_sentiment():
    

    data = request.files.get("file")
    if data == None:
        return 'No data (Use Postman to test the prediction)'
    else:
        prediction = predict.predict(data)

    output={
     "sentiment_analysis" : str(prediction)
    }
    return json.dumps(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)