import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def preprocess_data(data):
    data.fillna('', inplace=True)
    data['Transaction Remarks'] = data['Transaction Remarks'].str.lower()
    import re
    data['Transaction Remarks'] = data['Transaction Remarks'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return data

def split_and_train_model(data, algorithm='MNB'):
    X = data['Transaction Remarks']
    y = data['Category Expense']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    if algorithm == 'MNB':
        model = MultinomialNB()
    elif algorithm == 'SVM':
        model = SVC(kernel='linear', probability=True)
    elif algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid algorithm specified.")

    model.fit(X_train_features, y_train)

    y_pred_proba = model.predict_proba(X_test_features)
    confidence_scores = [max(probas) for probas in y_pred_proba] 

    test_data = pd.DataFrame({'Transaction Remarks': X_test, 'Predicted Category': model.predict(X_test_features), 'Confidence Score': confidence_scores})

    print("\nTested Data, Predicted Categories, and Confidence Scores:")
    print(test_data)

    joblib.dump({'model': model, 'vectorizer': vectorizer}, 'trainedModelV3.pkl')

    return model, vectorizer

data = pd.read_csv("dataset3.csv")
data = preprocess_data(data)
model, vectorizer = split_and_train_model(data, algorithm='MNB')

if model is not None:
    pass
