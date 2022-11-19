import gradio as gr
import numpy as np
from PIL import Image
import requests
import hopsworks
import hsml
import joblib

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

model = mr.get_model("titanic_modal", version=3)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def get_image(res):
    if res[0]==0:
        return "https://raw.githubusercontent.com/ayberkuckun/Serverless-ML/main/titanic/images/0.png"
        # return '../images/0.png'
    return "https://raw.githubusercontent.com/ayberkuckun/Serverless-ML/main/titanic/images/1.png"
    # return '../images/1.png'

def titanic(pclass, sex, age, sibsp, parch, fare, embarked):
    input_list = []
    final_inputs = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    input_list.append(embarked)
    # ['2', 'F', 3.0, 2.0, 3.0, 5.0, 'Q']
    for count, elem in enumerate(input_list):
        if count==0:
            final_inputs.append(int(elem))
        elif count==1:
            final_inputs.append(0 if elem == 'F' else 1)
        elif count==2 or count==3 or count==4 or count==5:
            final_inputs.append(elem)
        elif count==6:
            if elem=='C':
                final_inputs.extend((1, 0, 0))
            elif elem=='Q':
                final_inputs.extend((0, 1, 0))
            else:
                final_inputs.extend((0, 0, 1))

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(final_inputs).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    image_link = get_image(res)
    img = Image.open(requests.get(image_link, stream=True).raw)
    # img = Image.open(image_link)
    return img


demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analysis",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
# pclass, sex, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s
        gr.inputs.Radio(choices=['1', '2', '3'], label="Passenger Class"),
        gr.inputs.Radio(choices=['M', 'F'], label="Sex"),
        gr.inputs.Number(label="Age"),
        gr.inputs.Number(label="Number of Siblings/Spouses Aboard"),
        gr.inputs.Number(label="Number of Parents/Children Aboard"),
        gr.inputs.Number(label="Passenger Fare"),
        gr.inputs.Radio(choices=['C', 'Q', 'S'], label="Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()