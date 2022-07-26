import gradio as gr
from utils.constants import CSV_HEADER, NUMERIC_FEATURE_NAMES, NUMBER_INPUT_COLS
from utils.preprocess import create_max_values_map, create_dropdown_default_values_map, create_sample_test_data, CATEGORICAL_FEATURES_WITH_VOCABULARY
from utils.predict import batch_predict, user_input_predict

inputs_list = []
max_values_map = create_max_values_map()
dropdown_default_values_map = create_dropdown_default_values_map()
sample_input_df_val = create_sample_test_data()


demo = gr.Blocks()

with demo:
    
    gr.Markdown("# **Binary Classification using Gated Residual and Variable Selection Networks** \n")
    gr.Markdown("This space demonstrates the use of Gated Residual Networks (GRN) and Variable Selection Networks (VSN), proposed by Bryan Lim et al. in <a href=\"https://arxiv.org/abs/1912.09363/\">Temporal Fusion Transformers (TFT) for Interpretable Multi-horizon Time Series Forecasting</a> for structured data classification")
    gr.Markdown("Play around and see yourself 🤗 ")
    
    with gr.Tabs():
        
        with gr.TabItem("Predict using Example inputs"):
            gr.Markdown("**Input DataFrame** \n")
            input_df = gr.Dataframe(headers=CSV_HEADER,value=sample_input_df_val,)
            gr.Markdown("**Output DataFrame** \n")
            output_df = gr.Dataframe()
            gr.Markdown("**Make Predictions**")
            with gr.Row():
                compute_button = gr.Button("Predict")
        
        with gr.TabItem("Tweak inputs Yourself & Predict"):
            with gr.Tabs():
                
                with gr.TabItem("Numerical Inputs"):
                    gr.Markdown("Set values for numerical inputs here.")
                    for num_variable in NUMERIC_FEATURE_NAMES:
                        with gr.Column():
                            if num_variable in NUMBER_INPUT_COLS:
                                numeric_input = gr.Number(label=num_variable)
                            else:
                                curr_max_val = max_values_map["max_"+num_variable]
                                numeric_input = gr.Slider(0,curr_max_val, label=num_variable,step=1)
                            inputs_list.append(numeric_input)
                
                with gr.TabItem("Categorical Inputs"):
                    gr.Markdown("Choose values for categorical inputs here.")
                    for cat_variable in CATEGORICAL_FEATURES_WITH_VOCABULARY.keys():
                        with gr.Column():
                            categorical_input = gr.Dropdown(CATEGORICAL_FEATURES_WITH_VOCABULARY[cat_variable], label=cat_variable, value=str(dropdown_default_values_map["max_"+cat_variable]))
                            inputs_list.append(categorical_input)
                            
                    predict_button = gr.Button("Predict")
                    final_output = gr.Label()
    
    predict_button.click(user_input_predict, inputs=inputs_list, outputs=final_output)
    compute_button.click(batch_predict, inputs=input_df, outputs=output_df)
    gr.Markdown('\n Author: <a href=\"https://www.linkedin.com/in/shivalika-singh/\">Shivalika Singh</a> <br> Based on this <a href=\"https://keras.io/examples/structured_data/classification_with_grn_and_vsn/\">Keras example</a> by <a href=\"https://www.linkedin.com/in/khalid-salama-24403144/\">Khalid Salama</a> <br> Demo Powered by this <a href=\"https://huggingface.co/keras-io/structured-data-classification-grn-vsn/\">GRN-VSN model</a>')
    
    
demo.launch()