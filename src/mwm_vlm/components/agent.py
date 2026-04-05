import os
import json
import importlib
from dotenv import load_dotenv
import operator
from typing import Annotated, Literal, Sequence, TypedDict, Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI 
from openai import OpenAI

try:
    from . import tool_functions as t
except ImportError:
    t = importlib.import_module("tool_functions")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This is for LangGraph agent "brain"
LLM = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4",
    temperature=0
)

# This is for CrystallizationInterpreter
RAW_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 1. Define the State
# ==========================================
# ── Field contract ──────────────────────────────────────────────────────────
# confidence    : one of "high" | "medium" | "low"  (set by parse_result_node)
# ambiguity_flag: True  → the image is ambiguous and needs extra context
#                 False → the image is clear and can be classified directly
# All *_signal fields mirror the JSON output from extract_features_from_image.
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    The state of the agent. 
    'messages' uses `operator.add` to always append new messages to the existing list.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    image_path: str  # Store the image path to be accessible by all nodes
    crystal_signal: str
    precipitate_signal: str
    edge_signal: str
    phase_separation_signal: str
    skin_signal: str
    artifact_signal: str
    clarity_signal: str
    morphology_pattern: str
    ambiguity_flag: bool          # Contract: must be bool; True = needs extra context
    confidence: Literal["high", "medium", "low"]  # Contract: only these three values
    observation: str
    current_case_summary: str
    retrieved_cases: list[dict[str, Any]]
    final_report: str

AGENT_GRAPH_STRUCTURE_ROOT_PATH = "/Users/thuang/Documents/Personal/code/image-reasoning-ar26/research/agent_structure_graph"

CASE_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_parts",
    "case_library.json",
)
FINAL_OUTPUT_INSTRUCTION_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_parts",
    "final_output_instruction.txt",
)
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_CASES = 3

PARSED_FEATURE_DEFAULTS: dict[str, Any] = {
    "crystal_signal": "low",
    "precipitate_signal": "low",
    "edge_signal": "low",
    "phase_separation_signal": "low",
    "skin_signal": "low",
    "artifact_signal": "low",
    "clarity_signal": "low",
    "morphology_pattern": "unknown",
    "ambiguity_flag": True,
    "confidence": "low",
    "observation": "",
}


def _build_current_case_summary(state: AgentState) -> str:
    """Build a query text that mirrors the style of case['retrieval_text']."""
    return (
        f"{state.get('observation', '')}; "
        f"crystal_signal {state.get('crystal_signal', '')}; "
        f"precipitate_signal {state.get('precipitate_signal', '')}; "
        f"edge_signal {state.get('edge_signal', '')}; "
        f"phase_separation_signal {state.get('phase_separation_signal', '')}; "
        f"skin_signal {state.get('skin_signal', '')}; "
        f"artifact_signal {state.get('artifact_signal', '')}; "
        f"clarity_signal {state.get('clarity_signal', '')}; "
        f"morphology {state.get('morphology_pattern', '')}"
    )


def _embed_text(text: str) -> list[float]:
    response = RAW_CLIENT.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _cosine_similarity_score(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _load_case_library() -> list[dict[str, Any]]:
    with open(CASE_LIBRARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_final_output_instruction() -> str:
    with open(FINAL_OUTPUT_INSTRUCTION_PATH, "r", encoding="utf-8") as file:
        return file.read().strip()


def _build_final_output_context(state: AgentState) -> str:
    current_case_block = (
        "Current case:\n"
        f"- crystal_signal: {state.get('crystal_signal', '')}\n"
        f"- precipitate_signal: {state.get('precipitate_signal', '')}\n"
        f"- edge_signal: {state.get('edge_signal', '')}\n"
        f"- phase_separation_signal: {state.get('phase_separation_signal', '')}\n"
        f"- skin_signal: {state.get('skin_signal', '')}\n"
        f"- artifact_signal: {state.get('artifact_signal', '')}\n"
        f"- clarity_signal: {state.get('clarity_signal', '')}\n"
        f"- morphology_pattern: {state.get('morphology_pattern', '')}\n"
        f"- observation: {state.get('observation', '')}"
    )

    retrieved_cases = state.get("retrieved_cases", []) or []
    if not retrieved_cases:
        retrieved_block = "Retrieved cases:\nNone"
    else:
        retrieved_sections: list[str] = []
        for idx, case in enumerate(retrieved_cases, start=1):
            retrieved_sections.append(
                f"{idx}. {case.get('case_id', 'N/A')}\n"
                f"State: {case.get('state', 'N/A')}\n"
                f"Observation: {case.get('observation', 'N/A')}\n"
                f"What to do next: {case.get('what_to_do_next', 'N/A')}"
            )
        retrieved_block = "Retrieved cases:\n" + "\n\n".join(retrieved_sections)

    return f"{current_case_block}\n\n{retrieved_block}"


def _short_text(text: str, max_chars: int = 180) -> str:
    if not isinstance(text, str):
        return str(text)
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _compact_state_for_debug(state: dict[str, Any]) -> dict[str, Any]:
    """Return a concise, human-readable snapshot of the current AgentState."""
    messages = state.get("messages", []) or []
    last_message = messages[-1] if messages else None

    last_message_type = type(last_message).__name__ if last_message is not None else None
    last_message_preview = None
    if last_message is not None and hasattr(last_message, "content"):
        last_message_preview = _short_text(str(last_message.content), 220)

    retrieved_cases = state.get("retrieved_cases", []) or []
    retrieved_cases_preview = [
        {
            "case_id": c.get("case_id"),
            "state": c.get("state"),
            "score": c.get("score"),
        }
        for c in retrieved_cases[:TOP_K_CASES]
    ]

    return {
        "image_path": state.get("image_path"),
        "confidence": state.get("confidence"),
        "ambiguity_flag": state.get("ambiguity_flag"),
        "signals": {
            "crystal": state.get("crystal_signal"),
            "precipitate": state.get("precipitate_signal"),
            "edge": state.get("edge_signal"),
            "phase_separation": state.get("phase_separation_signal"),
            "skin": state.get("skin_signal"),
            "artifact": state.get("artifact_signal"),
            "clarity": state.get("clarity_signal"),
            "morphology": state.get("morphology_pattern"),
        },
        "observation": _short_text(state.get("observation", ""), 220),
        "current_case_summary": _short_text(state.get("current_case_summary", ""), 260),
        "retrieved_cases": retrieved_cases_preview,
        "final_report_preview": _short_text(state.get("final_report", ""), 260),
        "messages_count": len(messages),
        "last_message_type": last_message_type,
        "last_message_preview": last_message_preview,
    }


def _merge_state_update(running_state: dict[str, Any], update: dict[str, Any]) -> None:
    """Merge one node update into the running state shown in debug printouts."""
    for key, value in update.items():
        if key == "messages":
            running_state.setdefault("messages", [])
            running_state["messages"] = [*running_state["messages"], *value]
        else:
            running_state[key] = value


def _get_latest_tool_message(messages: Sequence[BaseMessage]) -> ToolMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg
    return None


def _build_parsed_feature_state(payload: dict[str, Any]) -> dict[str, Any]:
    # Keep only the feature fields we care about and fill missing ones with defaults.
    return {
        key: payload.get(key, default_value)
        for key, default_value in PARSED_FEATURE_DEFAULTS.items()
    }

# ==========================================
# 2. Bind Tools
# ==========================================
@tool
def extract_features_from_image_tool(image_path: str) -> str:
    """
    Tool description: 
    Use this tool to extract useful features for protein crystallization screening from a crystal microscope image.
    """
    return t.extract_features_from_image(image_path, RAW_CLIENT)

bound_llm = LLM.bind_tools([extract_features_from_image_tool])

# Tool registry dictionary
# This allows the graph network to map the LLM's invocation string back to the specific Python function
AVAILABLE_TOOLS = {
    "extract_features_from_image_tool": extract_features_from_image_tool,
}

# ==========================================
# 3. Define Nodes
# ==========================================
def agent_node(state: AgentState) -> dict[str, Any]:
    """
    The 'Brain' node. Calls the Vision Language Model (VLM).
    """
    messages = state["messages"]

    print("🧠 Agent Node: Thinking...")
    
    response = bound_llm.invoke(messages)
    
    if response.tool_calls:
        print(f"🤖 Agent decides to use tools: {response.tool_calls} \n")
    else:   
        print(f"🤖 Agent decides to output: {response.content} \n")
    
    return {"messages": [response]} 

def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """
    The 'Hand' node. Executes tools if requested by the VLM.
    """
    messages = state["messages"]
    last_message = messages[-1]
    image_path = state["image_path"]
    
    print("🛠️ Tool Node: Executing tools...")
    
    tool_messages = []
    
    # A real LLM might call multiple tools in parallel in a single response, so we need to iterate over them
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        print(f"   => Calling tool: {tool_name}, args: {tool_args}")
        
        # Sometimes the model might forget to pass the image path, we can protect against this by falling back to the global state's image_path
        if "image_path" not in tool_args:
             tool_args["image_path"] = image_path
             
        tool_instance = AVAILABLE_TOOLS.get(tool_name)
        if tool_instance:
            try:
                # Execute the corresponding LangChain Tool, passing in arguments
                result = tool_instance.invoke(tool_args)
            except Exception as e:
                result = f"Error executing tool: {e}"
        else:
            result = f"Error: Tool named {tool_name} not found"
            
        print(f"   => Tool result: {result}")
        
        # Core point: The tool execution result **must** be wrapped in a LangChain ToolMessage,
        # and returned with the corresponding tool_call_id. This is the only way to pass the state back to the brain (agent) so it recognizes the relationship.
        tool_messages.append(ToolMessage(
            content=str(result), 
            tool_call_id=tool_call_id
        ))
        
    return {"messages": tool_messages}

def parse_result_node(state: AgentState) -> dict[str, Any]:
    """
    The 'Parser' node. Reads the raw JSON string returned by the feature extraction
    tool and writes the structured fields into the agent state.

    Why a separate node?
    The tool returns a plain text string. Before any routing decision can be made,
    someone has to parse that string and promote it to typed state fields. Keeping
    this step in its own node means: if the tool output format changes, only this
    node needs updating — the routing logic stays untouched.
    """
    print("📋 Parse Node: Extracting structured fields from tool output...")

    last_tool_message = _get_latest_tool_message(state["messages"])
    if last_tool_message is None:
        print("⚠️  Parse Node: No tool message found. Defaulting to conservative values.")
        return PARSED_FEATURE_DEFAULTS.copy()

    try:
        data = json.loads(last_tool_message.content)
        parsed_state = _build_parsed_feature_state(data)
        print(
            f"✅ Parse Node: confidence={parsed_state.get('confidence')}, "
            f"ambiguity_flag={parsed_state.get('ambiguity_flag')}"
        )
        return parsed_state
    except json.JSONDecodeError as e:
        # Fail-safe: a parse failure routes to the cautious retrieve_cases branch
        print(f"⚠️  Parse Node: JSON parse failed ({e}). Defaulting to conservative values.")
        return PARSED_FEATURE_DEFAULTS.copy()


def final_output_node(state: AgentState) -> dict[str, Any]:
    """
    Generates the final reasoning output report.
    Supports both direct path and retrieve-then-final path.
    """
    print("📤 Final Output Node: generating final report...")

    try:
        instruction = _load_final_output_instruction()
        context_block = _build_final_output_context(state)
        final_prompt = (
            f"{instruction}\n\n"
            "Use the following structured context to produce the final report.\n\n"
            f"{context_block}"
        )

        response = LLM.invoke([HumanMessage(content=final_prompt)])
        final_report = str(response.content)

        print("✅ Final Output Node: final report generated.")
        print("\n🧾 ================== FINAL REPORT ==================")
        print(final_report)
        print("📘 ==================================================\n")

        return {
            "final_report": final_report,
            "messages": [response],
        }
    except Exception as e:
        error_report = f"Final output generation failed: {e}"
        print(f"⚠️  Final Output Node: {error_report}")
        print("\n🧾 ================== FINAL REPORT ==================")
        print(error_report)
        print("📘 ==================================================\n")
        return {
            "final_report": error_report,
            "messages": [AIMessage(content=error_report)],
        }


def retrieve_cases_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieves similar historical cases to support reasoning.
    Reached when confidence < 'high' or ambiguity_flag == True.
    Uses embedding + cosine similarity over case_library retrieval_text.
    """
    print("🔍 Retrieve Cases Node: Low-confidence / ambiguous path — retrieving cases...")

    current_case_summary = _build_current_case_summary(state)

    try:
        case_library = _load_case_library()
        query_embedding = _embed_text(current_case_summary)

        scored_cases: list[dict[str, Any]] = []
        for case in case_library:
            retrieval_text = case.get("retrieval_text", "")
            if not retrieval_text:
                continue

            case_embedding = _embed_text(retrieval_text)
            score = _cosine_similarity_score(query_embedding, case_embedding)
            scored_cases.append({"case": case, "score": score})

        top_k = sorted(scored_cases, key=lambda item: item["score"], reverse=True)[:TOP_K_CASES]
        top_cases = [
            {
                "score": round(item["score"], 4),
                "case_id": item["case"].get("case_id"),
                "state": item["case"].get("state"),
                "observation": item["case"].get("observation"),
                "what_to_do_next": item["case"].get("what_to_do_next"),
                "retrieval_text": item["case"].get("retrieval_text"),
            }
            for item in top_k
        ]

        retrieval_message = AIMessage(
            content=(
                "Retrieved top similar historical cases:\n"
                f"{json.dumps(top_cases, ensure_ascii=False, indent=2)}"
            )
        )

        return {
            "current_case_summary": current_case_summary,
            "retrieved_cases": top_cases,
            "messages": [retrieval_message],
        }
    except Exception as e:
        return {
            "current_case_summary": current_case_summary,
            "retrieved_cases": [],
            "messages": [AIMessage(content=f"Case retrieval failed: {e}")],
        }


# ==========================================
# 4. Define Edge Logic
# ==========================================
def route_by_confidence(state: AgentState) -> str:
    """
    Conditional edge function: decides the next node after parse_result_node.

    Why a conditional edge function and NOT a node?
    This function does zero computation — it only reads two already-populated
    state fields and returns a routing key. LangGraph uses that key to look up
    the next node in the edges map. Any actual work (parsing, calling an LLM,
    writing state) must live in a node; route functions are pure control-flow.
    """
    confidence = state.get("confidence", "low")
    ambiguity_flag = state.get("ambiguity_flag", True)

    if confidence == "high" and ambiguity_flag is False:
        print("🔀 Router: confidence=high, ambiguity=False → final_output")
        return "final_output"
    else:
        print(f"🔀 Router: confidence={confidence}, ambiguity={ambiguity_flag} → retrieve_cases")
        return "retrieve_cases"


def should_continue(state: AgentState) -> str:
    """
    Controls the flow. Checks if the agent wants to use a tool or finish.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last Message generated by the LLM contains a tool invocation (tool_calls) field
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # The model wants to use a tool, proceed to Action
        return "continue"
    else:
        # No tool invocation, the model gives the final conclusion, proceed to END
        return "end"

# ==========================================
# 5. Build and Compile the Graph
# ==========================================
def build_image_reasoning_agent():
    # Initialize the graph with our state definition
    workflow = StateGraph(AgentState)

    # --- Existing nodes ---
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_executor_node)

    # --- New nodes ---
    # parse_result: promotes the raw tool string into typed state fields
    # final_output / retrieve_cases: the two downstream reasoning branches
    workflow.add_node("parse_result", parse_result_node)
    workflow.add_node("final_output", final_output_node)
    workflow.add_node("retrieve_cases", retrieve_cases_node)

    # Set the starting point
    workflow.set_entry_point("agent")

    # agent → did LLM ask for a tool?
    #   yes ("continue") → action
    #   no  ("end")      → END  (edge case: agent decides not to use any tool)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )

    # action → parse_result
    # Previously this looped back to agent. Now we hand off to the parser
    # so that structured fields are available for the routing decision.
    workflow.add_edge("action", "parse_result")

    # parse_result → route_by_confidence decides the branch
    #   "final_output"  → final_output_node
    #   "retrieve_cases" → retrieve_cases_node
    workflow.add_conditional_edges(
        "parse_result",
        route_by_confidence,
        {
            "final_output": "final_output",
            "retrieve_cases": "retrieve_cases",
        }
    )

    # Both paths converge at final_output, which generates the reasoning report.
    # retrieve_cases feeds its results into final_output as additional context.
    workflow.add_edge("retrieve_cases", "final_output")
    workflow.add_edge("final_output", END)

    # Compile the state machine into an executable app
    app = workflow.compile()
    return app

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":

    from mwm_vlm.utils.common import save_agent_graph_structure

    DEBUG_PRINT_STATE = False
    SAVE_AGENT_GRAPH_STRUCTURE = False

    agent_app = build_image_reasoning_agent()
    
    if SAVE_AGENT_GRAPH_STRUCTURE:
        # Save the agent graph structure for iterative design
        save_agent_graph_structure(agent_app, AGENT_GRAPH_STRUCTURE_ROOT_PATH)

    # ⚠️ Please ensure there is a real test image at this path, or change it to an existing relative/absolute path
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "test", "1209.jpeg")
    
    # Build your initial prompt, since our tool implements the logic of "using the base model for image classification"
    # We directly let the Agent use the tool to provide the classification result
    initial_prompt = f"The image is located at this local path: '{test_image_path}'. Please use the extract_features_from_image_tool to extract features from it."
    
    # Initialize input state
    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)],
        "image_path": test_image_path
    }
    
    # Start the graph network!
    print("🚀 Starting LangGraph execution...")

    running_state: dict[str, Any] = {
        "messages": list(initial_state.get("messages", [])),
        "image_path": initial_state.get("image_path"),
    }
    
    # stream() returns the execution process step by step (think -> tool execution -> think -> end)
    for step_idx, s in enumerate(agent_app.stream(initial_state), start=1):
        print("\n--- Current step state snapshot ---")
        # Print the output of the current node
        for node_name, node_update in s.items():
            print(f"Step {step_idx} | Node '{node_name}'")
            if isinstance(node_update, dict):
                _merge_state_update(running_state, node_update)
                print(f"Updated keys: {list(node_update.keys())}")
            else:
                print("Node update is not a dict, skipped state merge.")

            if DEBUG_PRINT_STATE:
                print("AgentState summary:")
                print(json.dumps(_compact_state_for_debug(running_state), ensure_ascii=False, indent=2))
