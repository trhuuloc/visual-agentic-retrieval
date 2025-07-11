from supervisor.create_supervisor import build_graph
from retrieve.
def retrieve():
    image_path = "data/sample.jpg"
    question = "Có bao nhiêu người trong ảnh?"

    graph = build_graph()
    result = graph.invoke({
        "image_path": image_path,
        "question": question
    })

    print("Câu trả lời:", result["answer"])

if __name__ == "__main__":
    main()
