from utils.vision_utils import detect_objects, generate_caption

class VisualAgent:
    def __init__(self):
        pass

    def run(self, image_path):
        objects = detect_objects(image_path)
        caption = generate_caption(image_path)
        return {
            "objects": objects,
            "caption": caption
        }