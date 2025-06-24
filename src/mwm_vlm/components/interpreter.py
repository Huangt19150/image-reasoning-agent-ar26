import os
from PIL import Image

from mwm_vlm.components.prompt import (
    get_prompt_v1,
    get_prompt_v2,
)

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

    def classify_image(self):
        """
        Lv.1 Architecture: 
        This method uses a simple prompt to classify the image into one of the four categories:
        [clear, crystals, precipitate, other].
        """

        # Prepare the prompt
        prompts = get_prompt_v1(self.image_path)

        # Call the model
        response = self.openai_client.responses.create(
            model=self.llm_name,
            input=[
                {
                    "role": "user",
                    "content": prompts,
                }
            ],
            temperature=0.2,
        )

        return response.output_text

    def generate_image_caption(self):
        """
        Lv.2 Architecture: 
        Enriched prompt with context from protein crystallization result interpretation
        lab mannuls and examples.
        This method aims for a rich caption of the image covering as much morphology description
        as possible, instead of a over-simplified class label (e.g. crystal vs non-crystal).
        """

        # Prepare the prompt
        prompts = get_prompt_v2(self.image_path)

        # Call the model
        response = self.openai_client.responses.create(
            model=self.llm_name,
            input=[
                {
                    "role": "user",
                    "content": prompts,
                }
            ],
            temperature=0.2,
        )

        return response.output_text


if __name__ == "__main__":
    # Example usage
    from PIL import Image

    from openai import OpenAI

    OPENAI_API_KEY = ""  # Replace with your OpenAI API key
    CLIENT = OpenAI(api_key=OPENAI_API_KEY)

    # Load an image
    image_path = "/Users/thuang/Documents/Personal/code/microscopy-vlm-lpb25/app/examples/215.jpeg"
    image = Image.open(image_path)

    # Initialize the interpreter
    interpreter = CrystallizationInterpreter(
        openai_client=CLIENT,
        image=image,
        llm="gpt-4.1",
    )

    # Classify the image
    classification = interpreter.classify_image()
    print("Classification Result:", classification)
    """
    Example result: 
    Label: precipitate
    Explanation: The image shows a mostly clear droplet with several small, dark particulate spots scattered throughout, which are characteristic of precipitate formation rather than crystals or a completely clear solution.
    """

    # Generate caption
    # caption = interpreter.generate_image_caption()
    # print("Generated Caption:", caption)
    """
    Example caption:
    The drop is mostly clear (Score = 0) with no visible amorphous or gelatinous precipitate.Numerous small, dark, well-defined particles are dispersed throughout the drop, suggestive of microcrystals or salt crystals (approx. 5–20 µm in size), but further assessment under polarized light is needed to confirm birefringence and proteinaceous nature. There is no evidence of phase separation, oily films, or surface skin formation. No large crystals, spherulites, or needle bundles are observed. The drop is free of visible contaminants such as dust, fibers, or microbial growth. Illumination appears to be bright-field; no birefringence can be assessed from this image alone. Further validation under polarized light is recommended to distinguish between salt and protein microcrystals.
    """
    