import os
from datetime import datetime
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


def save_agent_graph_structure(agent_app, save_root_path: str):
    """
    Save the agent graph structure as a PNG image.
    """
    try:
        image_bytes = agent_app.get_graph().draw_mermaid_png()

        # Generate unique file name
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = os.path.join(save_root_path, f"agent_structure_{unique_name}.png")
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ The agent graph structure has been saved as {save_path}!")
    except Exception as e:
        print(f"⚠️ Could not generate PNG image, please check network or dependencies: {e}")
