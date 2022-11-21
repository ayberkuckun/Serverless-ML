import modal

LOCAL = True

if not LOCAL:
    stub = modal.Stub("titanic_daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])


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
                       "Embarked_C": [0],
                       "Embarked_Q": [0],
                       "Embarked_S": [0],
                       "Name_Capt": [0],
                       "Name_Col": [0],
                       "Name_Countess": [0],
                       "Name_Don": [0],
                       "Name_Dr": [0],
                       "Name_Jonkheer": [0],
                       "Name_Lady": [0],
                       "Name_Major": [0],
                       "Name_Master": [0],
                       "Name_Miss": [0],
                       "Name_Mlle": [0],
                       "Name_Mme": [0],
                       "Name_Mr": [0],
                       "Name_Mrs": [0],
                       "Name_Ms": [0],
                       "Name_Rev": [0],
                       "Name_Sir": [0],
                       "Cabin_A": [0],
                       "Cabin_B": [0],
                       "Cabin_C": [0],
                       "Cabin_D": [0],
                       "Cabin_E": [0],
                       "Cabin_F": [0],
                       "Cabin_G": [0],
                       "Cabin_T": [0],
                       "Cabin_U": [0]
                       })

    embarked = random.choice(['C', 'Q', 'S'])
    df[f'Embarked_{embarked}'] = [1]

    name = random.choice(["Capt", "Col", "Countess", "Don", "Dr", "Jonkheer", "Lady", "Major", "Master",
                          "Miss", "Mlle", "Mme", "Mr", "Mrs", "Ms", "Rev", "Sir"])
    df[f'Name_{name}'] = [1]

    cabin = random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'])
    df[f'Cabin_{cabin}'] = [1]

    return df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_synthetic_df = generate_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_synthetic_modal",
        version=1,
        primary_key=['Sex', 'Age', 'Ticket', 'FamilySize', "Fare", "Pclass"],
        description="Titanic synthetic dataset")
    titanic_fg.insert(titanic_synthetic_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
