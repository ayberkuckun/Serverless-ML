import gradio as gr
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/iris/images/latest_iris.png")
dataset_api.download("Resources/iris/images/actual_iris.png")
dataset_api.download("Resources/iris/images/df_recent.png")
dataset_api.download("Resources/iris/images/confusion_matrix.png")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predicted Image")
            input_img = gr.Image("latest_iris.png", elem_id="predicted-img")
        with gr.Column():
            gr.Label("Today's Actual Image")
            input_img = gr.Image("actual_iris.png", elem_id="actual-img")
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")

demo.launch()
