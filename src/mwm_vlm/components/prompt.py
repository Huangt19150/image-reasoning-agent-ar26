import os
import json
from mwm_vlm.utils.common import encode_image


PROMPT_PARTS_DIR = os.path.join(os.path.dirname(__file__), "prompt_parts")
EXAMPLE_IMAGES_DIR = os.path.join(PROMPT_PARTS_DIR, "example_images")
SUPPORTED_IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".webp")


def _find_example_image(example_id: str) -> str:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = os.path.join(EXAMPLE_IMAGES_DIR, f"{example_id}{suffix}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No matching image found for example '{example_id}' in {EXAMPLE_IMAGES_DIR}")


def _iter_example_pairs() -> list[tuple[str, str]]:
    example_pairs: list[tuple[str, str]] = []

    for file_name in sorted(os.listdir(EXAMPLE_IMAGES_DIR)):
        if not file_name.endswith("_output.json"):
            continue

        example_id = file_name.removesuffix("_output.json")
        json_path = os.path.join(EXAMPLE_IMAGES_DIR, file_name)
        image_path = _find_example_image(example_id)
        example_pairs.append((image_path, json_path))

    return example_pairs

def get_prompt(image_path):
    """
    Build the multimodal prompt for feature extraction.

    All examples are discovered automatically from prompt_parts/example_images.
    Add a new example by placing both:
    - <id>.<image extension>
    - <id>_output.json
    in that directory.
    """

    # Prepare the common part of the prompt
    with open(os.path.join(PROMPT_PARTS_DIR, "feature_extraction_instruction.txt"), "r", encoding="utf-8") as f:
        instructions = f.read()

    input_items = [{"type": "input_text", "text": instructions}]

    for index, (example_image_path, example_json_path) in enumerate(_iter_example_pairs(), start=1):
        with open(example_json_path, "r", encoding="utf-8") as f:
            example_output = json.load(f)

        input_items.extend([
            {"type": "input_text", "text": f"**Example {index}:**"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_image(example_image_path)}",
            },
            {"type": "input_text", "text": json.dumps(example_output, ensure_ascii=False)},
        ])

    input_items.extend([
        {"type": "input_text", "text": "**Now extract features from the following image:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
    ])

    return input_items
