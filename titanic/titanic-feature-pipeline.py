import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn", "sklearn", "dataframe-image"])
    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("./datasets/titanic_cleaned.csv")
    titanic_df.drop('Unnamed: 0', inplace=True, axis=1)
    titanic_df.rename(columns={'Embarked - C': 'Embarked_C', 'Embarked - Q': 'Embarked_Q',
                               'Embarked - S': 'Embarked_S'}, inplace=True)
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=2,
        primary_key=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked_C', 'Embarked_Q', 'Embarked_S'],
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})

if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
