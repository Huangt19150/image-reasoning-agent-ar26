import os
import gradio as gr

# NOTE: Deploy to hugging face space ONLY: Add the src directory to the Python path
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from callbacks import (
    run_agent_chat,
    run_agent_from_example,
    clear_all,
)
from ui_helpers import (
    _empty_features_html,
    _empty_cases_html,
)


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.css")
with open(CSS_PATH, "r", encoding="utf-8") as css_file:
    CSS = css_file.read()

example_images = [
    [os.path.join(EXAMPLES_DIR, f)]
    for f in sorted(os.listdir(EXAMPLES_DIR))
    if f.lower().endswith((".jpeg", ".jpg", ".png", ".gif", ".webp"))
]


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 😼 Schrödinger's Crystals 💠
        ## A MINI Proof-Of-Concept Protein Crystallization Image Reasoning Agent with GPT-5.4
        Code available in this repo: [image-reasoning-agent-ar26](https://github.com/Huangt19150/image-reasoning-agent-ar26)

        ## 🔷 Why Protein Crystals:
        Proteins are the tiny machines that keep all living things running, and understanding their shape is key to understanding how they work — and how to design new medicines.
        One powerful way to figure out a protein's 3D structure is by turning it into a crystal and analyzing it with X-ray crystallography. But here's the catch: proteins don't like to crystallize. 
        Scientists often need to run thousands of experiments just to grow one usable crystal.  
        To this end, MARCO (MAchine Recognition of Crystallization Outcomes) was established to bring real-world crystallization data to the machine learning community.
        The goal is to build smart tools that can automatically classify the outcomes of these tricky experiments. 
        By helping scientists spot crystals faster and more accurately, these tools can accelerate discoveries in biology and medicine.
        

        ## 📋 Quick Start Guide:
        1. Try one of the **Test Images** below to find the hidden crystals.
        2. (Optional) Classification result is cached to save cost on API request. Trust me 🤗 or click `🧹 Clear Cache` then observe the real latency of the prediction request (5-10s).
        3. Toggle the lists below to find out more insights!

        ## 📖 Reference:
        1. Dataset & Background: [MARCO](https://marco.ccr.buffalo.edu/about).
        """
    )

    image_path_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Agent Panel")
            agent_chatbot = gr.Chatbot(label="Agent Demo", height=500, sanitize_html=False, elem_id="agent-chatbot")
            chat_input = gr.MultimodalTextbox(
                placeholder="Attach an image and type your message, then press Enter...",
                file_types=["image"],
                file_count="single",
                submit_btn=True,
            )
            example_image_input = gr.Image(type="filepath", visible=False)
            gr.Examples(
                label="Example Images (click to auto-run)",
                examples=example_images,
                inputs=[example_image_input],
                cache_examples=False,
            )
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1, elem_classes=["right-panel", "evidence-panel"]):
            gr.Markdown("## Evidence Panel")
            image_viewer = gr.Image(type="pil", label="Current Image", interactive=False)
            features_view = gr.HTML(value=_empty_features_html(), elem_id="evidence-features")
            similar_cases_view = gr.HTML(value=_empty_cases_html(), elem_id="evidence-cases")

    submit_event = chat_input.submit(
        fn=run_agent_chat,
        inputs=[chat_input, agent_chatbot],
        outputs=[agent_chatbot, image_viewer, features_view, similar_cases_view],
    )
    submit_event.then(fn=lambda: None, inputs=None, outputs=[chat_input])

    example_image_input.change(
        fn=run_agent_from_example,
        inputs=[example_image_input],
        outputs=[agent_chatbot, image_viewer, features_view, similar_cases_view, image_path_state],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[agent_chatbot, chat_input, features_view, similar_cases_view, image_viewer, example_image_input, image_path_state],
    )

    # "How does it work?" Section
    gr.HTML("<hr style='border:0.5px solid #ccc; margin: 20px 0;'>")
    gr.Markdown("## 🔮 How Does It Work?")

    # with gr.Accordion(label="Prompt Design", open=False):
    #     with gr.Row():
    #         gr.Markdown("""
    #             ### Prompt Design  
    #             As you've noticed, this classifier is not a custom-trained CNN image classifier.
    #             Instead, it uses GPT-4, a Large Language Model with Vision and Reasoning capability, to classify images based on a prompt.
    #             Sepecifically, the prompt covers the following parts:
    #             - **Instructions:** A short task instruction with domain context and category definitions
    #             - **Examples:** 1 image from each of the 4 categories are provided as examples, along with an explanation to facilitate reasoning
    #             - **Test Image:** The actual test image, different from the example images, to be classified
    #         """)
            
    # with gr.Accordion(label="Batch Accuracy", open=False):
    #     with gr.Row():
    #         with gr.Column(scale=1):
    #             gr.Markdown("""
    #             ### Batch Accuracy
    #             🔶 **Note that this is only a very preliminary proof of concept.** 🔶  
    #             The current prompt was tested on a batch of **79** images (~20 each category), from a single source of image provider, with the following results:
    #             - **Accuracy:** 75%
    #             - **Precision:** 80%
    #             - **Recall:** 75%
    #             - **F1-score:** 75%
    #             - **Confusion Matrix:** (see on the right)
    #             - **Classification Report:**
    #             | category | precision | recall | f1-score | sample size |
    #             |---|---|---|---|---|
    #             | clear | 0.79 | 0.95 | 0.86 | 20 |
    #             | crystals | 0.93 | 0.68 | 0.79 | 19 |
    #             | other | 0.92 | 0.55 | 0.69 | 20 |
    #             | precipitate | 0.55 | 0.80 | 0.65 | 20 |
    #             | average | 0.80 | 0.75 | 0.75 | 79 |
    #             """)
            
    #         with gr.Column(scale=1):
    #             gr.Image(value="figures/confusion_matrix.png", label="Confusion Matrix", show_label=True)

    # with gr.Accordion(label="Cost & Tokens", open=False):
    #     with gr.Row():
    #         gr.Markdown("""
    #             ### Cost & Tokens  
    #             - **Cost:** One image classification with the current prompt design costs about **$0.01** (USD) using GPT-4.1. Batch job costs half the price.
    #             - **Tokens:** The current prompt design uses about **4000 tokens** (input + output) for each image classification.
    #         """)

    # with gr.Accordion(label="Why vLLM Could Be Helpful", open=False):
    #     with gr.Row():
    #         gr.Markdown("""
    #             ### Why vLLM Could Be Helpful  
    #             From this mini proof of concept, we can already observe 2 potential benefits of using vLLM:
    #             - **Small "Training" Set:** The current prompt design only uses 4 example images. More examples are going to be helpful, but the goal is to cover typical variations rather than providing huge learning bases.
    #             - **Language & Reasoning:** Language and reasoning provides a powerful handle to capture knowledge from domain experts using plain text and give examples in the prompt.
    #         """)

    # # "What's Next?" Section
    # gr.HTML("<hr style='border:0.5px solid #ccc; margin: 20px 0;'>")
    # gr.Markdown("## 🐾 What's Next?")

    # with gr.Accordion(label="Large Scale Evaluation", open=False):
    #     with gr.Row():
    #         gr.Markdown("""
    #             ### Large Scale Evaluation  
    #             The total number of test images provided by MARCO is 47,029, with the following major variations:
    #             - **Imaging System:** Images from different source organizations appears quite different in FOV size, droplet size, background color, etc.
    #             - **Category Definition:** The cateogry "other" has large variations by itself, and "crystals" also has different shapes, sometimes even indication of crystal growth.  
    #             Therefore evaluation on the whole test set is important to understand the model's performance in real world.
    #         """)

    # with gr.Accordion(label="Prompt Iteration", open=False):
    #     with gr.Row():
    #         gr.Markdown("""
    #             ### Prompt Iteration 
    #             Adding more images per category to the prompt to capture typical variations.
    #             - **5-10 Examples:** Considering **1M** context window, 5-10 example images per category is a reasonable target to start with.
    #         """)


if __name__ == "__main__":
    demo.launch(css=CSS)
