# import base64
import os
import sys
from PIL import Image
import gradio as gr
from openai import OpenAI

# NOTE: Deploy ONLY: Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mwm_vlm.utils.common import hash_image
from mwm_vlm.utils.gradio import load_cache, save_cache
from mwm_vlm.components.interpreter import CrystallizationInterpreter

# NOTE: Replace with actual OpenAI API key when test locally
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT = OpenAI(api_key=OPENAI_API_KEY)
CACHE = load_cache()


def clear_cache():
    """
    NOTE: Saved cache file `cache.json` will not be pushed to the File Repo when deployed on Hugging Face Spaces.
    Thus the global variable `CACHE` is the only control of cache when using the app on Hugging Face Spaces.
    This also means that the cache will be lost when the app is restarted (e.g. after `sleeping`).
    """
    global CACHE
    CACHE = {}
    save_cache(CACHE)
    print("Cache cleared.")
    return ["", ""]


def interpret(image: Image.Image) -> str:

    # Use cache if available
    hash_key = hash_image(image)
    if hash_key in CACHE:
        return CACHE[hash_key]

    interpreter = CrystallizationInterpreter(
        openai_client=CLIENT,
        image=image,
        llm="gpt-4.1",
    )
    classification_output = interpreter.classify_image()
    caption_output = interpreter.generate_image_caption()
    interpreter._cleanup_image()

    # Save the result to cache
    CACHE[hash_key] = [classification_output, caption_output]
    save_cache(CACHE)

    return [classification_output, caption_output]

# Example images for Gradio
example_images = [
    ["examples/215.jpeg"],
    ["examples/1401.jpeg"],
    ["examples/57465.jpeg"],
    ["examples/25.jpeg"]
]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 💠 Find the Crystals 
        ## A MINI Proof-Of-Concept Protein Crystallization Image Classifier using GPT4.
        Code available in this repo: [microscopy-vlm-lpb25](https://github.com/Huangt19150/microscopy-vlm-lpb25)
        
        🏆 Runner-up in [LPB25](https://www.kxsb.org/lpb25#winners) Imaging track!

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

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_classification = gr.Textbox(label="Classification Result")
            output_caption = gr.Textbox(label="Image Caption")
            clear_btn = gr.Button("Clear Result")
            clear_cache_btn = gr.Button("🧹 Clear Cache")

    submit_btn.click(fn=interpret, inputs=image_input, outputs=[output_classification, output_caption])
    clear_btn.click(fn=lambda: ["", ""], inputs=None, outputs=[output_classification, output_caption])
    clear_cache_btn.click(fn=clear_cache, outputs=[output_classification, output_caption])

    gr.Examples(
        label="Test Images You Can Try",
        examples=example_images,
        inputs=image_input,
        outputs=[output_classification, output_caption],
        cache_examples=False,
    )

    # "How does it work?" Section
    gr.HTML("<hr style='border:0.5px solid #ccc; margin: 20px 0;'>")
    gr.Markdown("## 🔮 How Does It Work?")

    with gr.Accordion(label="Prompt Design", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Prompt Design  
                As you've noticed, this classifier is not a custom-trained CNN image classifier.
                Instead, it uses GPT-4, a Large Language Model with Vision and Reasoning capability, to classify images based on a prompt.
                Sepecifically, the prompt covers the following parts:
                - **Instructions:** A short task instruction with domain context and category definitions
                - **Examples:** 1 image from each of the 4 categories are provided as examples, along with an explanation to facilitate reasoning
                - **Test Image:** The actual test image, different from the example images, to be classified
            """)
            
    with gr.Accordion(label="Batch Accuracy", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Batch Accuracy
                🔶 **Note that this is only a very preliminary proof of concept.** 🔶  
                The current prompt was tested on a batch of **79** images (~20 each category), from a single source of image provider, with the following results:
                - **Accuracy:** 75%
                - **Precision:** 80%
                - **Recall:** 75%
                - **F1-score:** 75%
                - **Confusion Matrix:** (see on the right)
                - **Classification Report:**
                | category | precision | recall | f1-score | sample size |
                |---|---|---|---|---|
                | clear | 0.79 | 0.95 | 0.86 | 20 |
                | crystals | 0.93 | 0.68 | 0.79 | 19 |
                | other | 0.92 | 0.55 | 0.69 | 20 |
                | precipitate | 0.55 | 0.80 | 0.65 | 20 |
                | average | 0.80 | 0.75 | 0.75 | 79 |
                """)
            
            with gr.Column(scale=1):
                gr.Image(value="figures/confusion_matrix.png", label="Confusion Matrix", show_label=True)

    with gr.Accordion(label="Cost & Tokens", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Cost & Tokens  
                - **Cost:** One image classification with the current prompt design costs about **$0.01** (USD) using GPT-4.1. Batch job costs half the price.
                - **Tokens:** The current prompt design uses about **4000 tokens** (input + output) for each image classification.
            """)

    with gr.Accordion(label="Why vLLM Could Be Helpful", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Why vLLM Could Be Helpful  
                From this mini proof of concept, we can already observe 2 potential benefits of using vLLM:
                - **Small "Training" Set:** The current prompt design only uses 4 example images. More examples are going to be helpful, but the goal is to cover typical variations rather than providing huge learning bases.
                - **Language & Reasoning:** Language and reasoning provides a powerful handle to capture knowledge from domain experts using plain text and give examples in the prompt.
            """)

    # "What's Next?" Section
    gr.HTML("<hr style='border:0.5px solid #ccc; margin: 20px 0;'>")
    gr.Markdown("## 🐾 What's Next?")

    with gr.Accordion(label="Large Scale Evaluation", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Large Scale Evaluation  
                The total number of test images provided by MARCO is 47,029, with the following major variations:
                - **Imaging System:** Images from different source organizations appears quite different in FOV size, droplet size, background color, etc.
                - **Category Definition:** The cateogry "other" has large variations by itself, and "crystals" also has different shapes, sometimes even indication of crystal growth.  
                Therefore evaluation on the whole test set is important to understand the model's performance in real world.
            """)

    with gr.Accordion(label="Prompt Iteration", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Prompt Iteration 
                Adding more images per category to the prompt to capture typical variations.
                - **5-10 Examples:** Considering **1M** context window, 5-10 example images per category is a reasonable target to start with.
            """)


if __name__ == "__main__":
    demo.launch()
