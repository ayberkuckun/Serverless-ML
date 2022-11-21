import modal

BACKFILL = False
LOCAL = True

if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn", "sklearn", "dataframe-image"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def generate_passenger():
    """
    Returns a single titanic passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"Survived": random.choice([0, 1]),
                       "Pclass": [random.choice([0, 1, 2])],
                       "Sex": [random.choice([0, 1])],
                       "Age": [random.uniform(0.0, 80.0)],
                       "Ticket": [random.randint(3, 18)],
                       "Fare": [random.uniform(0, 512)],
                       "Familysize": [random.randint(0, 10)],

                       "Embarked_C": [],
                       "Embarked_Q": [],
                       "Embarked_S": [],
                       "N": [random.uniform(petal_len_max, petal_len_min)],
                       "Cabin_A": [0],
                       "Cabin_B": [0],
                       "Cabin_C": [0],
                       "Cabin_D": [0],
                       "Cabin_E": [0],
                       "Cabin_F": [0],
                       "Cabin_G": [0],
                       "Cabin_T": [0],
                       "Cabin_U": [0],
                       })

    cabin = random.choice(['C', 'Q', 'S'])
    df[f'Cabin_{cabin}'] = [1]

    return df


def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df = generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0, 3)
    if pick_random >= 2:
        iris_df = virginica_df
        print("Virginica added")
    elif pick_random >= 1:
        iris_df = versicolor_df
        print("Versicolor added")
    else:
        iris_df = setosa_df
        print("Setosa added")

    return iris_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
    else:
        iris_df = get_random_iris_flower()

    iris_fg = fs.get_or_create_feature_group(
        name="iris_modal",
        version=1,
        primary_key=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        description="Iris flower datasets")
    iris_fg.insert(iris_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
