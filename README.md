# 🧠 Visual Agentic Retrieval

This project combines a **text-to-image retrieval system** (CLIP-based) with a **multi-agent visual question answering system**, coordinated using **LangGraph Supervisor**. After retrieving relevant images from text queries, users can ask follow-up questions about the selected image — and the supervisor routes the question to the correct agent (OCR, object detection, etc.).

---

## 🚀 Features

- 🔍 **Text-to-Image Retrieval** using OpenCLIP (contrastive learning)
- 🧠 **Visual Agents** for:
  - 🏷️ Text (OCR)
  - 📦 Object detection
- 🤖 **LangGraph Supervisor** to manage multi-agent routing
- 🧪 Easy-to-train + demo inference script
- 🖼️ Dataset: [Flickr30k]