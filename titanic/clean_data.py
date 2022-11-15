import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    df = pd.read_csv('titanic_raw.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df["Sex"] = np.where(df["Sex"] == "female", 0, 1)
    dataframe = df.copy()
    features = ['Embarked']
    # for example, features = ['Cars', 'Model, 'Year', ... ]
    for f in features:
        df = dataframe[[f]]
        df2 = (pd.get_dummies(df, prefix='', prefix_sep='')
               .max(level=0, axis=1)
               .add_prefix(f + ' - '))
        # the new feature names will be "<old_feature_name> - <categorical_value>"
        # for example, "Cars" will get transformed to "Cars - Minivan", "Cars - Truck", etc
        # add the new one-hot encoded column to the dataframe
        dataframe = pd.concat([dataframe, df2], axis=1)
        # you can remove the original columns, if you don't need them anymore (optional)
        dataframe = dataframe.drop([f], axis=1)
    dataframe.to_csv('titanic_cleaned.csv')
