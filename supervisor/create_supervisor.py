from langgraph_supervisor import create_supervisor
from my_agents import ocr_agent, object_agent
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model("gpt-4o"),
    agents=[ocr_agent, object_agent],
    prompt=(
        "You are a supervisor managing 2 agents:\n"
        "- OCR Agent: Handles reading text in the image.\n"
        "- Object Agent: Detects objects in the image.\n"
        "Decide which agent to call based on the question.\n"
        "Do not answer the question yourself.\n"
    ),
    output_mode="full_history",
).compile()