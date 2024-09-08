from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model_data = joblib.load('trainedModelV3.pkl')
model = model_data['model']
vectorizer = model_data['vectorizer']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    transaction_remark = data.get('TransactionRemark', '')

    remark = preprocess_data(pd.DataFrame({'Transaction Remarks': [transaction_remark]}))

    X = remark['Transaction Remarks']
    X_features = vectorizer.transform(X)
    y_pred = model.predict(X_features)[0]
    y_pred_proba = model.predict_proba(X_features)

    confidence_score = y_pred_proba.max()

    response = jsonify({'PredictedCategory': y_pred, 'ConfidenceScore': confidence_score})
    return response

def preprocess_data(data):
    data.fillna('', inplace=True)
    data['Transaction Remarks'] = data['Transaction Remarks'].str.lower()
    import re
    data['Transaction Remarks'] = data['Transaction Remarks'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return data

if __name__ == '__main__':
    app.run(debug=True)
