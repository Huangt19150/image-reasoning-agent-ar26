import html
import json
import base64
import mimetypes
import os


NODE_NAME_DISPLAY = {
    "input_router": "🧠 Understanding the request...",
    "image_observer": "🔍 Inspecting visual patterns...",
    "uncertainty_router": {
        "initial": "⚖️ Evaluating case uncertainty...",
        "confidence_high": "🟢 Uncertainty is low.",
        "confidence_medium": "🟡 Uncertainty is detected — route to check past cases...",
        "confidence_low": "🔴 Uncertainty is high — route to check past cases...",
    },
    "case_retriever": "📚 Consulting similar cases...",
    "report_generator": "🧾 Writing up findings...",
}

# Delay config follows the same node keys as NODE_NAME_DISPLAY.
SUMMARY_TRANSITION_DELAY_SEC = {
    "input_router": 0,
    "image_observer": 1,
    "uncertainty_router": {
        "initial": 5,
        "confidence_high": 2,
        "confidence_medium": 5,
        "confidence_low": 5,
    },
    "case_retriever": 5,
    "report_generator": 5,
}


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
    """Resolve UI-facing node labels, including confidence-specific uncertainty labels."""
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
    """Resolve per-node transition delay with confidence-aware uncertainty timing."""
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
    """Render summary text, with emphasis for uncertainty confidence states."""
    display_node_name = _resolve_node_display_name(node_name, node_update, prefer_initial=prefer_initial)
    safe_node_name = html.escape(display_node_name)

    confidence = str(node_update.get("confidence", "")).strip().lower()
    if node_name == "uncertainty_router" and not prefer_initial:
        if confidence == "high":
            return f"<span class='high-confidence'>{safe_node_name}</span>"
        if confidence == "medium":
            return f"<span class='medium-confidence'>{safe_node_name}</span>"
        if confidence == "low":
            return f"<span class='low-confidence'>{safe_node_name}</span>"
    return safe_node_name


def _format_stream_update(node_name: str, node_update: dict, prefer_initial: bool = False) -> str:
    lines = []
    has_tool_calls = False
    if node_update.get("final_report"):
        lines.append("Final report generated. Read below for details.")
    elif "messages" in node_update:
        for message in node_update["messages"]:
            if getattr(message, "tool_calls", None):
                has_tool_calls = True
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

    if node_name == "input_router" and not has_tool_calls and payload != "(no text payload)":
        return f"<div><strong>{summary_html}</strong><div>{safe_payload}</div></div>"

    return (
        f"<details>"
        f"<summary>{summary_html}</summary>"
        f"<div>{safe_payload}</div>"
        f"</details>"
    )


def _empty_features_html() -> str:
    return (
        "<div class='evidence-section'>"
        "<div class='evidence-title'>Extracted Features</div>"
        "<div class='evidence-empty'>No extracted features yet.</div>"
        "</div>"
    )


def _empty_cases_html() -> str:
    return (
        "<div class='evidence-section'>"
        "<div class='evidence-title'>Similar Cases</div>"
        "<div class='evidence-empty'>No similar cases yet.</div>"
        "</div>"
    )


def _extract_features_from_messages(messages) -> dict:
    for message in reversed(messages or []):
        content = str(getattr(message, "content", "")).strip()
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _render_features_html(features: dict) -> str:
    """Render selected extracted features in fixed order with value-based chip styling."""
    if not isinstance(features, dict) or not features:
        return _empty_features_html()

    ordered_feature_fields = [
        ("morphology_pattern", "Morphology Pattern"),
        ("crystal_signal", "Crystal Signal"),
        ("precipitate_signal", "Precipitate Signal"),
        ("edge_signal", "Edge Signal"),
        ("phase_separation_signal", "Phase Separation Signal"),
        ("skin_signal", "Skin Signal"),
        ("ambiguity_flag", "Ambiguity Flag"),
        ("confidence", "Confidence"),
    ]

    def chip_class(key: str) -> str:
        if key == "morphology_pattern":
            return "chip chip-shape"
        if key in {"ambiguity_flag", "confidence"}:
            return "chip chip-conf"
        return "chip chip-positive"

    def value_class(key: str, value) -> str:
        lowered = str(value).strip().lower()
        if key == "ambiguity_flag":
            if lowered == "true":
                return " ambiguity-true"
            if lowered == "false":
                return " ambiguity-false"
            return ""
        if key == "confidence" and lowered == "low":
            return " confidence-low"
        if lowered == "high":
            return " high"
        if lowered == "medium":
            return " medium"
        return ""

    chips = []
    for key, label_text in ordered_feature_fields:
        if key not in features:
            continue
        safe_label = html.escape(label_text)
        safe_value = html.escape(str(features.get(key, "")))
        safe_value_class = value_class(key, features.get(key, ""))
        chips.append(
            f"<span class='{chip_class(key)}'><strong>{safe_label}</strong>"
            f"<span class='chip-value{safe_value_class}'>{safe_value}</span></span>"
        )

    if not chips:
        return _empty_features_html()

    return (
        "<div class='evidence-section'>"
        "<div class='evidence-title'>Extracted Features</div>"
        f"<div class='feature-chip-list'>{''.join(chips)}</div>"
        "</div>"
    )


def _render_cases_html(cases: list[dict]) -> str:
    """Render top retrieved cases as lightweight evidence cards."""
    if not isinstance(cases, list) or not cases:
        return _empty_cases_html()

    def humanize_state(raw_state) -> str:
        text = str(raw_state or "N/A").strip()
        if not text:
            return "N/A"
        return " ".join(text.replace("_", " ").split()).title()

    cards = []
    for case in cases[:3]:
        case_id = html.escape(str(case.get("case_id", "N/A")))
        state = html.escape(humanize_state(case.get("state", "N/A")))
        score = html.escape(str(case.get("score", "N/A")))
        observation = html.escape(str(case.get("observation", "")))
        next_step = html.escape(str(case.get("what_to_do_next", "")))
        cards.append(
            "<div class='case-card'>"
            f"<div class='case-card-header'><span class='case-id'>{case_id}</span><span class='case-pattern'>{state} · score {score}</span></div>"
            f"<p class='case-pattern'><strong>Observation:</strong> {observation}</p>"
            f"<p class='case-action'><strong>Next Step:</strong> {next_step}</p>"
            "</div>"
        )

    return (
        "<div class='evidence-section'>"
        "<div class='evidence-title'>Similar Cases</div>"
        f"<div class='case-card-list'>{''.join(cards)}</div>"
        "</div>"
    )


def _format_final_report_markdown(report: dict) -> str:
    """Render final report markdown and color-code confidence using shared chip styles."""
    confidence_raw = str(report.get("confidence", "")).strip()
    confidence_value = confidence_raw.lower()
    confidence_class = ""
    if confidence_value == "high":
        confidence_class = " high"
    elif confidence_value == "medium":
        confidence_class = " medium"
    elif confidence_value == "low":
        confidence_class = " confidence-low"

    confidence_display = html.escape(confidence_raw)
    if confidence_class:
        confidence_display = f"<span class='chip-value{confidence_class}'>{confidence_display}</span>"

    return f"""
**Observation**  
{report.get("observation", "")}

**Possible interpretation**  
{report.get("possible_interpretation", "")}

**Supporting context**  
{report.get("supporting_context", "")}

**Final Confidence**  
{confidence_display}

**Recommended next step**  
{report.get("recommended_next_step", "")}
""".strip()


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}
