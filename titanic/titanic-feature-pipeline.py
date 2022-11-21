import modal
import pandas as pd

LOCAL = True

if not LOCAL:
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
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1"
                             "/titanic.csv")
    titanic_clean_df = clean_data(titanic_df)
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_last_modal",
        version=1,
        primary_key=['Sex', 'Age', 'Ticket', 'FamilySize', "Fare", "Pclass"],
        description="Titanic clean dataset")
    titanic_fg.insert(titanic_clean_df, write_options={"wait_for_job": False})


def clean_data(df):
    df.Embarked = df.Embarked.fillna(df.Embarked.mode())
    df.Cabin = df.Cabin.fillna('Unknown')
    df.Cabin = df.Cabin.map(lambda x: x[0])
    df.Name = df.Name.apply(lambda x: x.split('.')[0].split(',')[1].split(' ')[-1].strip())
    grouped = df.groupby(['Name', 'Sex'], group_keys=False)
    df.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
    df.Sex = df.Sex.map({"female": 0, "male": 1})
    df.Ticket = df.Ticket.map(lambda x: len(x))
    df["FamilySize"] = df.SibSp + df.Parch
    df = df.drop(["PassengerId", "SibSp", "Parch"], axis=1)

    features = ['Embarked', "Name", "Cabin"]
    for f in features:
        numerical_col = pd.get_dummies(df[f], prefix=f)
        df = pd.concat([df, numerical_col], axis=1)
        df = df.drop([f], axis=1)

    return df


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        with stub.run():
            f()
