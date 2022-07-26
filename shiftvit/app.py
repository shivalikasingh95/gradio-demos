import gradio as gr
from utils.predict import predict, predict_batch
import os

inputs_list = []


demo = gr.Blocks()

# sample_1 = ['examples/ship.png']
# sample_2 = ['examples/deer.jpg']
# sample_image = gr.Image(type='filepath')
# examples = gr.components.Dataset(components=[sample_image], samples=[sample_1, sample_2], type='values')
# with gr.Column():
            #     examples.render()
            #     examples.click(load_example, examples, input_images)
# def load_example(image):
#     return image[0]
    
with demo:
    
    gr.Markdown("# **<p align='center'>ShiftViT: A Vision Transformer without Attention</p>**")
    gr.Markdown("This space demonstrates the use of ShiftViT proposed in the paper: <a href=\"https://arxiv.org/abs/2201.10801/\">When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism</a> for image classification task.")
    gr.Markdown("Vision Transformers (ViTs) have proven to be very useful for Computer Vision tasks. Many researchers believe that the attention layer is the main reason behind the success of ViTs.")
    gr.Markdown("In the ShiftViT paper, the authors have tried to show that the attention mechanism may not be vital for the success of ViTs by replacing the attention operation with a shifting operation.")
    
    with gr.Tabs():
        
        with gr.TabItem("Skip Uploading!"):
            
            gr.Markdown("Just click *Run Model* below:")
            with gr.Box():
                gr.Markdown("**Prediction Probabilities** \n")
                output_df = gr.Dataframe(headers=["image","1st_highest_probability", "2nd_highest_probability","3rd_highest_probability"],datatype=["str", "str", "str", "str"])
                gr.Markdown("**Output Plot** \n")
                output_plot = gr.Image(type='filepath')

            gr.Markdown("**Predict**")
            
            with gr.Box():
                with gr.Row():
                    compute_button = gr.Button("Run Model")
                
        
        with gr.TabItem("Upload & Predict"):
            with gr.Box():
                
                with gr.Row():
                    input_image = gr.Image(type='filepath',label="Input Image", show_label=True)
                    output_label = gr.Label(label="Model", show_label=True)
            
            gr.Markdown("**Predict**")
            
            with gr.Box():
                with gr.Row():
                    submit_button = gr.Button("Submit")
            
            gr.Markdown("**Examples:**")
            gr.Markdown("The model is trained to classify images belonging to the following classes:")

            with gr.Column():
                gr.Examples("examples/set2", [input_image], output_label, predict, cache_examples=True)
        
    
    compute_button.click(predict_batch, inputs=input_image, outputs=[output_plot,output_df])
    submit_button.click(predict, inputs=input_image, outputs=output_label)
    
    gr.Markdown('\n Author: <a href=\"https://www.linkedin.com/in/shivalika-singh/\">Shivalika Singh</a> <br> Based on this <a href=\"https://keras.io/examples/vision/shiftvit/\">Keras example</a> by <a href=\"https://twitter.com/ariG23498\">Aritra Roy Gosthipaty</a> and <a href=\"https://twitter.com/ritwik_raha\">Ritwik Raha</a> <br> Demo Powered by this <a href=\"https://huggingface.co/shivi/shiftvit/\">ShiftViT model</a>')
    
demo.launch()
