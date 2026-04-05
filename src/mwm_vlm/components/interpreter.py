import os
import json
from PIL import Image

from mwm_vlm.components.prompt import get_prompt


SIGNAL_LEVELS = {"low", "medium", "high"}
MORPHOLOGY_PATTERNS = {
    "none",
    "unknown",
    "amorphous-like",
    "needle-like",
    "plate-like",
    "block-like",
    "cluster-like",
    "microcrystalline-like",
    "spherulite-like",
}

# Keep this mapping explicit rather than guessing from SDK internals.
# Add or update values when you switch models and know their limits.
MODEL_INPUT_TOKEN_LIMITS = {
    "gpt-5.4": 272000, # 1M token is allowed but under 272k is cheaper
    "gpt-4.1": None,
}
DEBUG_FLAG = True


def _feature_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "crystal_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "precipitate_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "edge_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "phase_separation_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "skin_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "artifact_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "clarity_signal": {"type": "string", "enum": ["low", "medium", "high"]},
            "morphology_pattern": {
                "type": "string",
                "enum": [
                    "none",
                    "unknown",
                    "amorphous-like",
                    "needle-like",
                    "plate-like",
                    "block-like",
                    "cluster-like",
                    "microcrystalline-like",
                    "spherulite-like",
                ],
            },
            "ambiguity_flag": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "observation": {"type": "string"},
        },
        "required": [
            "crystal_signal",
            "precipitate_signal",
            "edge_signal",
            "phase_separation_signal",
            "skin_signal",
            "artifact_signal",
            "clarity_signal",
            "morphology_pattern",
            "ambiguity_flag",
            "confidence",
            "observation",
        ],
    }


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _as_level(value, default: str = "low") -> str:
    value = str(value).strip().lower()
    return value if value in SIGNAL_LEVELS else default


def _as_morphology(value, default: str = "unknown") -> str:
    value = str(value).strip().lower()
    return value if value in MORPHOLOGY_PATTERNS else default


def _as_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _normalize_feature_payload(payload: dict) -> dict:
    return {
        "crystal_signal": _as_level(payload.get("crystal_signal", "low")),
        "precipitate_signal": _as_level(payload.get("precipitate_signal", "low")),
        "edge_signal": _as_level(payload.get("edge_signal", "low")),
        "phase_separation_signal": _as_level(payload.get("phase_separation_signal", "low")),
        "skin_signal": _as_level(payload.get("skin_signal", "low")),
        "artifact_signal": _as_level(payload.get("artifact_signal", "low")),
        "clarity_signal": _as_level(payload.get("clarity_signal", "low")),
        "morphology_pattern": _as_morphology(payload.get("morphology_pattern", "unknown")),
        "ambiguity_flag": _as_bool(payload.get("ambiguity_flag", True), default=True),
        "confidence": _as_level(payload.get("confidence", "low")),
        "observation": str(payload.get("observation", "")).strip(),
    }


def _format_token_limit(limit: int | None) -> str:
    if limit is None:
        return "not configured"
    return f"{limit:,}"


def _print_prompt_token_usage(model_name: str, response) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    model_limit = MODEL_INPUT_TOKEN_LIMITS.get(model_name)

    print("\n📏 ================= PROMPT TOKEN USAGE =================")
    print(f"Model: {model_name}")
    print(f"Prompt tokens (actual): {input_tokens if input_tokens is not None else 'unknown'}")
    print(f"Model input token limit: {_format_token_limit(model_limit)}")
    if input_tokens is not None and model_limit:
        usage_ratio = input_tokens / model_limit
        print(f"Prompt usage ratio: {usage_ratio:.2%}")
    print(f"Output tokens: {output_tokens if output_tokens is not None else 'unknown'}")
    print(f"Total tokens: {total_tokens if total_tokens is not None else 'unknown'}")
    print("📐 =====================================================\n")

class CrystallizationInterpreter:
    """
    This class is used to interpret Crystallization droplet results using
    LLMs (Large Language Models) based methods. Different prompts cover different
    architectures and can be used in comparison.
    """

    def __init__(
            self,
            openai_client,
            image=None,
            llm="gpt-4.1",
        ):
        # OpenAI client
        self.openai_client = openai_client
        self.llm_name = llm

        # Image to be interpreted
        self._dump_image(image)


    def _dump_image(self, image: Image.Image, image_savepath="input_image.png"):
        """
        Save a input image (e.g. uploaded from the frontend) to a local path.
        """
        self.image = image
        self.image_path = image_savepath
        image.save(image_savepath)
    
    def _cleanup_image(self):
        """
        Clean up the local image file if needed.
        """
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        self.image_path = None

    def extract_features_from_image(self):
        """
        This method uses LLMs to extract features from the image.
        """

        # Prepare the prompt
        prompts = get_prompt(self.image_path)

        # Call the model
        response = self.openai_client.responses.create(
            model=self.llm_name,
            input=[
                {
                    "role": "user",
                    "content": prompts,
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "crystallization_features",
                    "strict": True,
                    "schema": _feature_schema(),
                }
            },
            temperature=0,
        )

        if DEBUG_FLAG:
            _print_prompt_token_usage(self.llm_name, response)

        parsed = json.loads(_strip_code_fence(response.output_text))
        return _normalize_feature_payload(parsed)


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv
    from PIL import Image
    from openai import OpenAI

    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLIENT = OpenAI(api_key=OPENAI_API_KEY)

    # Load an image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "test", "451193.png")
    image = Image.open(image_path)

    # Initialize the interpreter
    interpreter = CrystallizationInterpreter(
        openai_client=CLIENT,
        image=image,
        llm="gpt-5.4",
    )

    print(json.dumps(interpreter.extract_features_from_image(), indent=4))
    