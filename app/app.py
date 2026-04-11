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

CHAT_INPUT_PLACEHOLDER = "Attach an image (message is optional), then press Enter..."
CHAT_INPUT_LOCKED_PLACEHOLDER = "Analysis complete. Click 'Clear' to start another run, or try an example image."


def _chat_input_state_after_example_change(example_path):
    if example_path:
        return gr.update(
            value=None,
            interactive=False,
            placeholder=CHAT_INPUT_LOCKED_PLACEHOLDER,
        )
    return gr.update(
        value=None,
        interactive=True,
        placeholder=CHAT_INPUT_PLACEHOLDER,
    )


# Gradio Interface
with gr.Blocks() as demo:
    gr.Image(value="figures/logo.png", height=180, show_label=False)

    with gr.Row():
        gr.Markdown(
            """
            ### A Proof-of-Concept Image Reasoning Agent for Protein Crystallization (powered by GPT-5.4) [Code Repo](https://github.com/Huangt19150/image-reasoning-agent-ar26)
            """)
        
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## 🤖 What This Agent Does
                
                1. extracts visual features  
                2. estimates uncertainty  
                3. retrieves similar cases (if needed)  
                4. outputs a structured report with next-step recommendations  

                **Not just "what is this?" — but "what should I do next?"**
                """)

        with gr.Column():
            gr.Markdown(
                """
                ## 😼 Why "Schrödinger's Crystal"?

                Crystallization images are often ambiguous — not clearly “crystal” or “not crystal”.

                The sample can be **both — until we reason about it**.

                This agent is designed to resolve that **uncertainty** and support **decisions**.
                """)

        with gr.Column():
            gr.Markdown(
                """
                ## 💠 Why Protein Crystals?

                Protein crystallization is key for structure and drug discovery,  
                but experiments are large-scale, noisy, and hard to interpret.

                Datasets like [MARCO](https://marco.ccr.buffalo.edu/about) enable automated analysis —  
                this project explores going further with **reasoning agents**.
                """)
            
    with gr.Row():
        gr.Markdown(
            """
            ## 📋 Quick Start

            Try a **Test Image** or upload your own → the agent analyzes and recommends actions. ↓ Expand **How It Works** for agent graph and knowledge base.

            """)

    image_path_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Agent Panel")
            agent_chatbot = gr.Chatbot(label="Agent Demo", height=500, sanitize_html=False, elem_id="agent-chatbot")
            chat_input = gr.MultimodalTextbox(
                placeholder=CHAT_INPUT_PLACEHOLDER,
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
    submit_event.then(
        fn=lambda: gr.update(
            value=None,
            interactive=False,
            placeholder=CHAT_INPUT_LOCKED_PLACEHOLDER,
        ),
        inputs=None,
        outputs=[chat_input],
    )

    example_change_event = example_image_input.change(
        fn=run_agent_from_example,
        inputs=[example_image_input],
        outputs=[agent_chatbot, image_viewer, features_view, similar_cases_view, image_path_state],
    )
    example_change_event.then(
        fn=_chat_input_state_after_example_change,
        inputs=[example_image_input],
        outputs=[chat_input],
    )

    clear_event = clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[agent_chatbot, chat_input, features_view, similar_cases_view, image_viewer, example_image_input, image_path_state],
    )
    clear_event.then(
        fn=lambda: gr.update(interactive=True, placeholder=CHAT_INPUT_PLACEHOLDER),
        inputs=None,
        outputs=[chat_input],
    )

    # "How does it work?" Section
    gr.HTML("<hr style='border:0.5px solid #ccc; margin: 20px 0;'>")
    gr.Markdown("## 🔮 How Does It Work?")

    with gr.Accordion(label="Agent Design - The Reasoning Graph", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                    ### The Reasoning Graph
                    
                    Under the hood, the agent is built as a reasoning graph.
                    
                    It first extracts features, then estimates uncertainty.
                            
                    If uncertainty is low, it makes a direct decision.
                            
                    If uncertainty is high, it retrieves similar cases and performs case-based reasoning before producing a recommendation.
                    """)
            with gr.Column(scale=2):
                gr.Image(value="figures/agent_structure.png", label="Agent Structure", show_label=True, height=400)
    
    with gr.Accordion(label="Knowledge Base", open=False):
        with gr.Row():
            gr.Markdown("""
                Crystallization interpretation reference cases are extracted from this publically available handbook: [Crystal Growth 101](https://hamptonresearch.com/uploads/cg_pdf/CG101_COMPLETE_2019.pdf)
            """)

    with gr.Accordion(label="Cost & Tokens", open=False):
        with gr.Row():
            gr.Markdown("""
                ### Cost & Tokens  
                One agent round costs an average of **$0.016** (USD), 10k tokens, using GPT-5.4.
            """)

if __name__ == "__main__":
    demo.launch(css=CSS)
