import gradio as gr
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/titanic/images/latest_titanic.png")
dataset_api.download("Resources/titanic/images/actual_titanic.png")
dataset_api.download("Resources/titanic/images/df_recent.png")
dataset_api.download("Resources/titanic/images/confusion_matrix.png")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Passenger's Predicted Outcome")
            input_img = gr.Image("latest_titanic.png", elem_id="predicted-img")
        with gr.Column():
            gr.Label("Passenger's Actual Outcome")
            input_img = gr.Image("actual_titanic.png", elem_id="actual-img")
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")

demo.launch()
