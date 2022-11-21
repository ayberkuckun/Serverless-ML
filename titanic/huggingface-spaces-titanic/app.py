import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

one_hot_title_dict = {"Capt": 0, "Col": 0, "Countess": 0, "Don": 0, "Dr": 0, "Jonkheer": 0, "Lady": 0, "Major": 0,
                      "Master": 0, "Miss": 0, "Mlle": 0, "Mme": 0, "Mr": 0, "Mrs": 0, "Ms": 0, "Rev": 0, "Sir": 0,
                      }
one_hot_cabin_dict = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "T": 0, "U": 0}
one_hot_embark_dict = {"C": 0, "Q": 0, "S": 0}


def get_image(res):
    if res == 0:
        return "https://raw.githubusercontent.com/ayberkuckun/Serverless-ML/main/titanic/images/0.png"

    else:
        return "https://raw.githubusercontent.com/ayberkuckun/Serverless-ML/main/titanic/images/1.png"


def get_title_feature(name):
    one_hot_title_dict[name] = 1

    return list(one_hot_title_dict.values())


def get_sex_feature(sex):
    if sex == 'F':
        sex = 0
    else:
        sex = 1

    return sex


def get_cabin_feature(cabin):
    one_hot_cabin_dict[cabin] = 1

    return list(one_hot_cabin_dict.values())


def get_embarked_feature(embark):
    one_hot_embark_dict[embark] = 1

    return list(one_hot_embark_dict.values())


def titanic(pclass, title, sex, age, sibsp, parch, ticket, fare, cabin, embarked):
    input_list = []
    input_list.append(int(pclass))
    input_list.append(get_sex_feature(sex))
    input_list.append(age)
    input_list.append(int(ticket))
    input_list.append(fare)
    input_list.append(sibsp + parch)
    input_list.extend(get_embarked_feature(embarked))
    input_list.extend(get_title_feature(title))
    input_list.extend(get_cabin_feature(cabin))

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    image_link = get_image(res[0])
    img = Image.open(requests.get(image_link, stream=True).raw)

    return img


demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analysis",
    description="Experiment with different passengers to see if they may survive.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Radio(choices=['1', '2', '3'], label="Passenger Class"),
        gr.inputs.Dropdown(choices=['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master',
                                    'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'], label="Title"),
        gr.inputs.Radio(choices=['M', 'F'], label="Sex"),
        gr.inputs.Slider(minimum=0.0, maximum=80.0, label="Age"),
        gr.inputs.Number(label="Number of Siblings/Spouses Aboard"),
        gr.inputs.Number(label="Number of Parents/Children Aboard"),
        gr.inputs.Dropdown(choices=['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '15', '16', '17', '18'], label="Ticket Code"),
        gr.inputs.Slider(minimum=0.0, maximum=512.0, label="Passenger Fare"),
        gr.inputs.Dropdown(choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], label="Cabin Code (U = Unknown)"),
        gr.inputs.Dropdown(choices=['C', 'Q', 'S'], label="Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

