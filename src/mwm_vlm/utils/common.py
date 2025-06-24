import base64
import hashlib
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def hash_image(image: Image.Image) -> str:
    """
    Generate a SHA-256 hash of the image file.
    This can be used to uniquely identify the image for caching purposes.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()
