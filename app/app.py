import os
import uuid
import html
import time
import json
from PIL import Image
import gradio as gr
from langchain_core.messages import HumanMessage

from mwm_vlm.components.agent import build_image_reasoning_agent


AGENT_APP = build_image_reasoning_agent()
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_uploads")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

# Map internal LangGraph node names to UI labels shown in stream summaries.
NODE_NAME_DISPLAY = {
    "agent": "🧠 Understanding the image...",
    "action": "🔍 Inspecting visual patterns...",
    "parse_result": {
        "initial": "⚖️ Evaluating case ambiguity...",
        "confidence_high": "🟢 Ambiguity is low.",
        "confidence_medium": "🟡 Ambiguity is detected — route to check past cases...",
        "confidence_low": "🔴 Ambiguity is high — route to check past cases...",
    },
    "retrieve_cases": "📚 Consulting similar cases...",
    "final_output": "🧾 Writing up findings...",
}

SUMMARY_TRANSITION_DELAY_SEC = {
    "agent": 0,
    "action": 1,
    "parse_result": {
        "initial": 5,
        "confidence_high": 2,
        "confidence_medium": 5,
        "confidence_low": 5,
    },
    "retrieve_cases": 5,
    "final_output": 5,
}

CSS = """
#agent-chatbot .agent-intro {
    margin-bottom: 6px;
}

#agent-chatbot details {
    margin: 4px 0 !important;
}

#agent-chatbot details:not(:first-of-type) {
  margin-top: 6px !important;
}

#agent-chatbot summary {
  cursor: pointer;
  line-height: 1.4;
}

#agent-chatbot details > div {
  margin-top: 4px;
  padding-left: 14px;
  white-space: pre-wrap;
}

#agent-chatbot .medium-confidence {
  color: #d97706;
  font-weight: 600;
  background: rgba(217,119,6,0.08);
  padding: 2px 6px;
  border-radius: 6px;
}

#agent-chatbot .high-confidence {
    color: #15803d;
    font-weight: 600;
    background: rgba(21,128,61,0.08);
    padding: 2px 6px;
    border-radius: 6px;
}

#agent-chatbot .low-confidence {
    color: #b91c1c;
    font-weight: 600;
    background: rgba(185,28,28,0.08);
    padding: 2px 6px;
    border-radius: 6px;
}

#agent-chatbot .final-divider {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid #e5e7eb;
  font-weight: 600;
}
"""

example_images = [
    [os.path.join(EXAMPLES_DIR, f)]
    for f in sorted(os.listdir(EXAMPLES_DIR))
    if f.lower().endswith((".jpeg", ".jpg", ".png", ".gif", ".webp"))
]


def _save_uploaded_image(image: Image.Image) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"upload_{uuid.uuid4().hex}.png")
    image.save(path)
    return path


def _build_agent_prompt(image_path: str, user_text: str) -> str:
    instruction = (
        f"The image is located at this local path: '{image_path}'. "
        "Please use the extract_features_from_image_tool to extract features from this image first, "
        "then provide the final report."
    )
    user_text = (user_text or "").strip()
    if user_text:
        return f"{instruction}\n\nAdditional user request: {user_text}"
    return instruction


def _resolve_node_display_name(node_name: str, node_update: dict, prefer_initial: bool = False) -> str:
    mapped = NODE_NAME_DISPLAY.get(node_name, node_name)
    if not isinstance(mapped, dict):
        return str(mapped)

    if prefer_initial:
        return str(mapped.get("initial", node_name))

    confidence = str(node_update.get("confidence", "")).strip().lower()
    if confidence in {"high", "medium", "low"}:
        return str(mapped.get(f"confidence_{confidence}", mapped.get("initial", node_name)))
    return str(mapped.get("initial", node_name))


def _resolve_summary_transition_delay_sec(
    node_name: str,
    node_update: dict,
    prefer_initial: bool = False,
) -> float:
    mapped = SUMMARY_TRANSITION_DELAY_SEC.get(node_name, 0)
    if not isinstance(mapped, dict):
        try:
            return max(float(mapped), 0.0)
        except (TypeError, ValueError):
            return 0.0

    if prefer_initial:
        key = "initial"
    else:
        confidence = str(node_update.get("confidence", "")).strip().lower()
        key = f"confidence_{confidence}" if confidence in {"high", "medium", "low"} else "initial"

    try:
        return max(float(mapped.get(key, 0)), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _message_to_stream_text(message) -> str:
    """Convert a LangChain message to readable stream text."""
    content = str(getattr(message, "content", "")).strip()
    if content:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return content

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        tool_lines = []
        for call in tool_calls:
            if isinstance(call, dict):
                tool_name = call.get("name", "unknown_tool")
                tool_args = call.get("args", {})
                tool_lines.append(f"Tool call -> {tool_name}: {tool_args}")
            else:
                tool_lines.append(f"Tool call -> {call}")
        return "\n".join(tool_lines)

    return ""


def _state_update_to_stream_text(node_update: dict) -> str:
    """Render node state delta as readable text for stream fallback."""
    if not isinstance(node_update, dict):
        return ""

    state_delta = {
        key: value
        for key, value in node_update.items()
        if key not in {"messages", "final_report"}
    }
    if not state_delta:
        return ""

    return "State update:\n" + json.dumps(state_delta, ensure_ascii=False, indent=2, default=str)


def _format_summary_html(node_name: str, node_update: dict, prefer_initial: bool = False) -> str:
    """Render summary text, with optional emphasis for specific confidence states."""
    display_node_name = _resolve_node_display_name(node_name, node_update, prefer_initial=prefer_initial)
    safe_node_name = html.escape(display_node_name)

    confidence = str(node_update.get("confidence", "")).strip().lower()
    if node_name == "parse_result" and not prefer_initial:
        if confidence == "high":
            return f"<span class='high-confidence'>{safe_node_name}</span>"
        if confidence == "medium":
            return f"<span class='medium-confidence'>{safe_node_name}</span>"
        if confidence == "low":
            return f"<span class='low-confidence'>{safe_node_name}</span>"
    return safe_node_name


def _format_stream_update(node_name: str, node_update: dict, prefer_initial: bool = False) -> str:
    lines = []
    if node_update.get("final_report"):
        lines.append("Final report generated. Read below for details.")
    elif "messages" in node_update:
        for message in node_update["messages"]:
            content = _message_to_stream_text(message)
            if content:
                lines.append(content)

    if len(lines) == 0:
        state_text = _state_update_to_stream_text(node_update)
        if state_text:
            lines.append(state_text)

    if len(lines) == 0:
        lines.append("(no text payload)")

    payload = "\n\n".join(lines) if lines else "(no text payload)"
    safe_payload = html.escape(payload)
    summary_html = _format_summary_html(node_name, node_update, prefer_initial=prefer_initial)

    return (
        f"<details>"
        f"<summary>{summary_html}</summary>"
        f"<div>{safe_payload}</div>"
        f"</details>"
    )


def _stream_agent(image_path: str, user_text: str):
    prompt = _build_agent_prompt(image_path, user_text)
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "image_path": image_path,
    }
    stream_text = "<div class='agent-intro'>🤖 Agent started...</div>"
    final_report = ""
    yield stream_text, final_report
    for step_idx, step_output in enumerate(AGENT_APP.stream(initial_state), start=1):
        for node_name, node_update in step_output.items():
            if isinstance(node_update, dict):
                if node_name == "parse_result":
                    initial_block = _format_stream_update(
                        node_name,
                        node_update={},
                        prefer_initial=True,
                    )
                    stream_text += initial_block
                    yield stream_text, final_report
                    initial_delay = _resolve_summary_transition_delay_sec(
                        node_name,
                        node_update={},
                        prefer_initial=True,
                    )
                    if initial_delay > 0:
                        time.sleep(initial_delay)

                    parsed_block = _format_stream_update(node_name, node_update)
                    stream_text = stream_text.replace(initial_block, parsed_block, 1)
                    if node_update.get("final_report"):
                        final_report = node_update["final_report"]
                    yield stream_text, final_report
                    parsed_delay = _resolve_summary_transition_delay_sec(node_name, node_update)
                    if parsed_delay > 0:
                        time.sleep(parsed_delay)
                else:
                    stream_text += _format_stream_update(node_name, node_update)
                    node_delay = _resolve_summary_transition_delay_sec(node_name, node_update)
                    pending_final_report = node_update.get("final_report")

                    if pending_final_report:
                        yield stream_text, final_report
                        if node_delay > 0:
                            time.sleep(node_delay)
                        final_report = pending_final_report
                        yield stream_text, final_report
                    else:
                        yield stream_text, final_report
                        if node_delay > 0:
                            time.sleep(node_delay)


def _format_final_report_markdown(report: dict) -> str:
    return f"""
**Observation**  
{report.get("observation", "")}

**Possible interpretation**  
{report.get("possible_interpretation", "")}

**Supporting context**  
{report.get("supporting_context", "")}

**Confidence**  
{report.get("confidence", "")}

**Recommended next step**  
{report.get("recommended_next_step", "")}
""".strip()


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def run_agent_chat(message, history):
    """Triggered by MultimodalTextbox submit (Enter or click send).
    message: {"text": str, "files": [filepath_str, ...]}
    history: list of {"role", "content"} dicts (Gradio 6 messages format)
    """
    user_text = (message.get("text") or "").strip() if isinstance(message, dict) else ""
    files = message.get("files", []) if isinstance(message, dict) else []

    image_path = ""
    pil_image = None
    if files:
        raw = files[0] if isinstance(files[0], str) else (files[0].get("path") or "")
        if raw:
            pil_image = Image.open(raw)
            pil_image.load()
            image_path = _save_uploaded_image(pil_image)

    history = list(history or [])

    if not image_path:
        history.append(_msg("user", user_text or "(no input)"))
        history.append(_msg("assistant", "⚠️ Please attach an image first."))
        yield history, None
        return

    user_display = user_text if user_text else f"[Image: {os.path.basename(image_path)}]"
    if user_text:
        user_display += f"\n\n[Image: {os.path.basename(image_path)}]"
    history.append(_msg("user", user_display))
    history.append(_msg("assistant", ""))

    final_report_appended = False
    for stream_text, streamed_report in _stream_agent(image_path, user_text):
        if not final_report_appended:
            history[-1] = _msg("assistant", stream_text)
        if isinstance(streamed_report, dict) and not final_report_appended:
            final_report_md = _format_final_report_markdown(streamed_report)
            history.append(
                _msg(
                    "assistant",
                    f"<div class='final-divider'></div>\n\n## 📄 Final Report\n\n{final_report_md}",
                )
            )
            final_report_appended = True
        yield history, pil_image


def run_agent_from_example(example_path: str):
    example_path = (example_path or "").strip()
    if not example_path:
        history = []
        history.append(_msg("assistant", "No example image loaded."))
        yield history, None, ""
        return

    message = {"text": "", "files": [example_path]}
    for updated_history, pil_image in run_agent_chat(message, []):
        yield updated_history, pil_image, example_path


def clear_all():
    # agent_chatbot, chat_input, image_viewer, example_image_input, image_path_state
    return [], None, None, None, ""


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
        with gr.Column(scale=1):
            gr.Markdown("## Agent Panel")
            agent_chatbot = gr.Chatbot(label="Agent Demo", height=500, sanitize_html=False, elem_id="agent-chatbot")
            chat_input = gr.MultimodalTextbox(
                placeholder="Attach an image and type your message, then press Enter...",
                file_types=["image"],
                file_count="single",
                submit_btn=True,
            )
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("## Evidence Panel")
            image_viewer = gr.Image(type="pil", label="Current Image", interactive=False)
            example_image_input = gr.Image(type="filepath", visible=False)
            gr.Examples(
                label="Example Images (click to auto-run)",
                examples=example_images,
                inputs=[example_image_input],
                cache_examples=False,
            )

    submit_event = chat_input.submit(
        fn=run_agent_chat,
        inputs=[chat_input, agent_chatbot],
        outputs=[agent_chatbot, image_viewer],
    )
    submit_event.then(fn=lambda: None, inputs=None, outputs=[chat_input])

    example_image_input.change(
        fn=run_agent_from_example,
        inputs=[example_image_input],
        outputs=[agent_chatbot, image_viewer, image_path_state],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[agent_chatbot, chat_input, image_viewer, example_image_input, image_path_state],
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
