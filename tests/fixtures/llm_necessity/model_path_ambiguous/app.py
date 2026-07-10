import gradio as gr

# Model path is ambiguous - could be ./models or ./model
MODEL_PATH = "./models"

def load_model():
    """Load model from local path."""
    import os
    if os.path.exists(MODEL_PATH):
        return f"Model loaded from {MODEL_PATH}"
    return "Model not found"

def predict(text):
    model_info = load_model()
    return f"{model_info}: {text}"

demo = gr.Interface(fn=predict, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
