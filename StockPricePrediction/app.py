import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained pipeline-
with open("stockPrice_knn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


# 2. Logic Function
def predict_stock(stock1, stock2, stock3, stock4, stock5):

    input_df = pd.DataFrame([[
        stock1, stock2, stock3, stock4, stock5
    ]],
    columns=['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5'])
    
    # Predict target
    prediction = model.predict(input_df)[0]
    
    return f"Predicted Stock_2 Price : {prediction:.2f}"


# 3. Gradio Interface
inputs = [
    gr.Number(label="Stock_1"),
    gr.Number(label="Stock_2"),
    gr.Number(label="Stock_3"),
    gr.Number(label="Stock_4"),
    gr.Number(label="Stock_5"),
]

app = gr.Interface(
    fn=predict_stock,
    inputs=inputs,
    outputs="text",
    title="Stock Price Predictor",
    description="Predict Stock_2 price based on Stock_1–4 and Unnamed:0 features."
)


# 4. Launch App
app.launch(share=True)
