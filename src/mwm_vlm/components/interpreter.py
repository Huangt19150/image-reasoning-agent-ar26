import os
from PIL import Image

from mwm_vlm.components.prompt import get_prompt_v1

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
    image_path = "/Users/thuang/Documents/Personal/code/image-reasoning-ar26/data/other_source/73836793958d18b9e365d8d8120e6d7a.JPG"
    image = Image.open(image_path)

    # Initialize the interpreter
    interpreter = CrystallizationInterpreter(
        openai_client=CLIENT,
        image=image,
        llm="gpt-5.4",
    )

    print(interpreter.extract_features_from_image())
    