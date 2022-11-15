import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    df = pd.read_csv('titanic_cleaned.csv').drop(['Unnamed: 0'], axis=True)
    y = df.Survived
    X = df.drop(['Survived'], axis=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(X_train, y_train)
    preds = xgb_cl.predict(X_test)
    print(accuracy_score(y_test, preds))