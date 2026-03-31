# 🚀 DeepSearch AI – Search Inside Videos Using AI

## 🧠 Problem

Massive volumes of video footage from CCTV and surveillance systems remain underutilized due to the lack of efficient search mechanisms. Current systems rely on manual scanning or basic metadata, making it difficult to locate specific events within hours of footage. This creates delays in critical areas such as law enforcement, security, and insurance investigations.

---

## 💡 Solution

DeepSearch AI is a multimodal AI-powered video search engine that allows users to search video content using natural language. By leveraging vision-language models, the system converts both video frames and user queries into comparable embeddings, enabling accurate retrieval of relevant timestamps where specific events occur.

---

## 🎯 MVP Scope

For this hackathon, we developed a working prototype that processes short video clips (30–60 seconds), extracts frames at intervals, and allows users to search using plain English queries. The system returns matching timestamps along with preview frames in near real-time.

---

## ⚙️ Tech Stack

- Python  
- OpenCV  
- CLIP (Multimodal AI Model)  
- NumPy  
- Streamlit (UI)

---

## 🔁 Working Flow

1. Upload video  
2. Extract frames at intervals  
3. Convert frames into embeddings  
4. Convert user query into embedding  
5. Compute cosine similarity  
6. Return matching timestamps with preview frames  

---

## ⚡ Features

- Natural language video search  
- Frame-level semantic matching  
- Real-time results  
- Timestamp-based retrieval  
- Preview frame display  

---

## 🎥 Demo

👉 Add your demo video link here (Loom / YouTube)

---

## 📸 Screenshots

Add images inside `/screenshots` folder:

- Upload screen  
- Search interface  
- Results with timestamps  
- Frame previews  

---

## 💻 Installation & Setup

```bash
git clone https://github.com/your-username/DeepSearch-AI.git
cd DeepSearch-AI
pip install -r requirements.txt
