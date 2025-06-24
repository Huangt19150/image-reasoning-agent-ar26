import os
from mwm_vlm.utils.common import encode_image

def get_prompt_v2(image_path):
    """
    For image captioning.
    """
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_parts/captioning_instruction.txt"),
        "r") as file:
        instructions = file.read()

    # Base64 encode the test image
    base64_test = encode_image(image_path)

    # Prepare the input items for the prompt
    input_items = [
        {"type": "input_text", "text": instructions},
        # Now the actual test image
        {"type": "input_text", "text": "**Now generate caption for the following image:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_test}"},
    ]

    return input_items

def get_prompt_v1(image_path):
    """
    This is the original prompt used in the first version of the app.
    """

    # Base64 encode each example image and your test image
    base64_example1 = encode_image(os.path.join(os.path.dirname(__file__), "prompt_parts/example_images_v1/6488.jpeg"))
    base64_example2 = encode_image(os.path.join(os.path.dirname(__file__), "prompt_parts/example_images_v1/220.jpeg"))
    base64_example3 = encode_image(os.path.join(os.path.dirname(__file__), "prompt_parts/example_images_v1/200.jpeg"))
    base64_example4 = encode_image(os.path.join(os.path.dirname(__file__), "prompt_parts/example_images_v1/9123.jpeg"))
    base64_test = encode_image(image_path)

    # Prepare the common part of the prompt
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_parts/classification_instruction.txt"),
        "r") as f:
        instructions = f.read()

    # Build the list of input items
    input_items = [
        {"type": "input_text", "text": instructions},

        # Example 1
        {"type": "input_text", "text": "**Example 1:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example1}"},
        {"type": "input_text", "text": "Label: crystals\nExplanation: Distinct crystal structures can be seen forming in the droplet."},

        # Example 2
        {"type": "input_text", "text": "**Example 2:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example2}"},
        {"type": "input_text", "text": "Label: clear\nExplanation: The droplet is transparent and shows no visible particles or structures."},

        # Example 3
        {"type": "input_text", "text": "**Example 3:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example3}"},
        {"type": "input_text", "text": "Label: precipitate\nExplanation: Granular particles are scattered across the droplet indicating precipitation."},

        # Example 4
        {"type": "input_text", "text": "**Example 4:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example4}"},
        {"type": "input_text", "text": "Label: other\nExplanation: The image shows unexpected structures which are not clear solution, crystals, or precipitate."},

        # Now the actual test image
        {"type": "input_text", "text": "**Now classify the following image:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_test}"},
    ]

    return input_items
