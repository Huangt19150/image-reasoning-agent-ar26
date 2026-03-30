from PIL import Image
from openai import OpenAI
from interpreter import CrystallizationInterpreter


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
    classification = interpreter.extract_features_from_image()

    return classification

# Your available tools dictionary
tools_map = {
    "extract_features_from_image": extract_features_from_image
}
