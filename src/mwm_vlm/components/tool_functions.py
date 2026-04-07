from PIL import Image
import json
import importlib
from openai import OpenAI

try:
    from .interpreter import CrystallizationInterpreter
except ImportError:
    CrystallizationInterpreter = importlib.import_module("interpreter").CrystallizationInterpreter


def extract_features_from_image(image_path: str, client: OpenAI) -> str:
    """
    Use this tool to extract useful features for protein crystallization screening from a crystal microscope image.
    """

    # Load an image
    image = Image.open(image_path)

    # Initialize the interpreter
    interpreter = CrystallizationInterpreter(
        openai_client=client,
        image=image,
        llm="gpt-5.4",
    )

    # Extract features from the image
    features = interpreter.extract_features_from_image()

    return json.dumps(features, ensure_ascii=False)
