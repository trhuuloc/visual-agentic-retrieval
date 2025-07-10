from typing import Annotated
from langchain_core.tools import tool
from ultralytics import YOLO
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

yolo_model = YOLO("yolo11x.pt")

class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Image path or URL")


class ObjectDetectingAndCountingInput(BaseModel):
    text: str = Field(description="Path or URL to the image in the format PNG or JPG/JPEG")

@tool(
    "detect_and_count_objects",
    description="Detect and count objects within the image. The return will be a dictionary, containing the counting dictionary (counting how many instance of each object class) and a list of dictionaries, containing the object names, confidence scores, and location in the image (in (x1, x2, y1, y2) format).",
    args_schema=ObjectDetectingAndCountingInput
)
def detect_and_count_object_tool(
    text: Annotated[str, "Path or URL to the image"], llm
):
    """Detect objects within the image using YOLOv11 model"""

    try:
        parser = PydanticOutputParser(pydantic_object=ImageInput)

        prompt = PromptTemplate.from_template(
            "Extract the image path or URL from the following input:\n\n{input}\n\n{format_instructions}"
        ).partial(format_instructions=parser.get_format_instructions())
        extractor_chain = prompt | llm | parser
        parsed: ImageInput = extractor_chain.invoke({"input": text})
    except Exception as e:
        return f"Failed to extract image URL: {str(e)}"

    image_path_or_url = parsed.image_path_or_url
    if not image_path_or_url:
        return "No image URL found in the input."

    results = yolo_model(image_path_or_url, verbose=False)

    detections = []
    counting = {}

    # Process each result
    for result in results:
        boxes = result.boxes
        class_names = result.names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })

            counting[class_name] = counting.get(class_name, 0) + 1

    return str({'counting': counting, 'detections': detections})