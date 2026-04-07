import os
import time
import uuid

from PIL import Image
from langchain_core.messages import HumanMessage

from mwm_vlm.components.agent import build_image_reasoning_agent
from ui_helpers import (
    _build_agent_prompt,
    _resolve_summary_transition_delay_sec,
    _format_stream_update,
    _empty_features_html,
    _empty_cases_html,
    _extract_features_from_messages,
    _render_features_html,
    _render_cases_html,
    _format_final_report_markdown,
    _msg,
)


AGENT_APP = build_image_reasoning_agent()
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_uploads")


def _save_uploaded_image(image: Image.Image) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"upload_{uuid.uuid4().hex}.png")
    image.save(path)
    return path


def _stream_agent(image_path: str, user_text: str):
    prompt = _build_agent_prompt(image_path, user_text)
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "image_path": image_path,
    }
    stream_text = "<div class='agent-intro'>🤖 Agent started...</div>"
    final_report = ""
    features_html = _empty_features_html()
    pending_features_html = features_html
    cases_html = _empty_cases_html()
    yield stream_text, final_report, features_html, cases_html
    for step_output in AGENT_APP.stream(initial_state):
        for node_name, node_update in step_output.items():
            if isinstance(node_update, dict):
                if node_name == "action":
                    extracted_features = _extract_features_from_messages(node_update.get("messages", []))
                    if extracted_features:
                        pending_features_html = _render_features_html(extracted_features)
                elif node_name == "retrieve_cases":
                    cases_html = _render_cases_html(node_update.get("retrieved_cases", []))

                if node_name == "parse_result":
                    initial_block = _format_stream_update(
                        node_name,
                        node_update={},
                        prefer_initial=True,
                    )
                    stream_text += initial_block
                    yield stream_text, final_report, features_html, cases_html
                    initial_delay = _resolve_summary_transition_delay_sec(
                        node_name,
                        node_update={},
                        prefer_initial=True,
                    )
                    if initial_delay > 0:
                        time.sleep(initial_delay)

                    features_html = pending_features_html
                    parsed_block = _format_stream_update(node_name, node_update)
                    stream_text = stream_text.replace(initial_block, parsed_block, 1)
                    if node_update.get("final_report"):
                        final_report = node_update["final_report"]
                    yield stream_text, final_report, features_html, cases_html
                    parsed_delay = _resolve_summary_transition_delay_sec(node_name, node_update)
                    if parsed_delay > 0:
                        time.sleep(parsed_delay)
                else:
                    stream_text += _format_stream_update(node_name, node_update)
                    node_delay = _resolve_summary_transition_delay_sec(node_name, node_update)
                    pending_final_report = node_update.get("final_report")

                    if pending_final_report:
                        yield stream_text, final_report, features_html, cases_html
                        if node_delay > 0:
                            time.sleep(node_delay)
                        final_report = pending_final_report
                        yield stream_text, final_report, features_html, cases_html
                    else:
                        yield stream_text, final_report, features_html, cases_html
                        if node_delay > 0:
                            time.sleep(node_delay)


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
        yield history, None, _empty_features_html(), _empty_cases_html()
        return

    user_display = user_text if user_text else f"[Image: {os.path.basename(image_path)}]"
    if user_text:
        user_display += f"\n\n[Image: {os.path.basename(image_path)}]"
    history.append(_msg("user", user_display))
    history.append(_msg("assistant", ""))

    final_report_appended = False
    for stream_text, streamed_report, features_html, cases_html in _stream_agent(image_path, user_text):
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
        yield history, pil_image, features_html, cases_html


def run_agent_from_example(example_path: str):
    example_path = (example_path or "").strip()
    if not example_path:
        history = []
        history.append(_msg("assistant", "No example image loaded."))
        yield history, None, _empty_features_html(), _empty_cases_html(), ""
        return

    message = {"text": "", "files": [example_path]}
    for updated_history, pil_image, features_html, cases_html in run_agent_chat(message, []):
        yield updated_history, pil_image, features_html, cases_html, example_path


def clear_all():
    return [], None, _empty_features_html(), _empty_cases_html(), None, None, ""
