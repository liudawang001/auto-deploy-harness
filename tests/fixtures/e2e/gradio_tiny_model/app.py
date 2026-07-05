import gradio as gr


def predict(text):
    return "web_result:" + text


demo = gr.Interface(fn=predict, inputs="text", outputs="text")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

