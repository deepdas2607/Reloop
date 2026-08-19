<div align="center">
  
# 🧬 ReLoop: Object Detection for Waste Segregation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/Frontend-React_Vite-cyan)](https://react.dev/)
[![Qdrant](https://img.shields.io/badge/Vector_Database-Qdrant-red)](https://qdrant.tech/)
[![Edge AI](https://img.shields.io/badge/AI-Offline_Edge-green)]()

> **"Turning Liability (Waste) into Asset (Feedstock)."**

</div>

---

## 🌍 The Problem

The Circular Economy fails due to **Information Asymmetry**.

- **Waste Pickers** sit on piles of valuable raw material ("Trash") but don't know who buys it.
- **Factories** pay premiums for virgin material because they can't locate clean, segregated scrap.

**ReLoop** bridges this gap. It is an **Agentic AI Engine** that audits waste composition in real-time, learns from human feedback, and performs a semantic handshake with industrial procurement manifests.

---

## 📊 The TACO Dataset Advantage

Unlike generic object detection models trained on standard datasets (like COCO), ReLoop's vision engine is explicitly fine-tuned on the **TACO (Trash Annotations in Context)** dataset. 

- **15 Specific Waste Classes:** We mapped TACO down to highly specific industrial categories like *Crisp MLP Packet*, *Styrofoam*, and *Hazardous Batteries*.
- **Combating Class Imbalance:** Real-world waste is highly imbalanced (e.g., thousands of plastic bottles, but very few batteries). We implemented custom loss geometries (**Focal Loss, CIoU, and DFL**) into our YOLOv11m training loop to actively counteract this, ensuring high accuracy even on rare, difficult objects.

---

## 📸 System Architecture (Multi-Agent)

ReLoop operates via a Hub-and-Spoke orchestration of specialized AI Agents:

| **Agent**           | **Role**              | **Tech Stack**                                                                               |
| :------------------ | :-------------------- | :------------------------------------------------------------------------------------------- |
| **👀 Urban Miner**  | Perception & Analysis | **YOLOv11m** fine-tuned on **TACO** (Detection) + **Florence-2** (Captioning) + **SigLIP** (Material Classification) |
| **👂 Audio Agent**  | Intent & Override     | **OpenAI Whisper 2026** (Speech-to-Text) with Translation                                    |
| **🧠 Memory Agent** | Episodic Recall       | **Qdrant** (Vector DB) storing past user corrections & material knowledge                    |
| **🏭 Market Agent** | Demand Matching       | **FastEmbed** + **PyMuPDF** to parse factory PDFs and match vectors                          |

---

## 🚀 Key Features

### 1. Tri-Layer Perception Engine

We don't rely on a single model. ReLoop uses a **Neural-Symbolic Cascade**:

- **Layer 1:** `YOLOv11m` **fine-tuned on TACO**. We implement specialized loss configurations to ensure high-fidelity bounding box regression and classification.
- **Layer 2:** `SigLIP` classifies material properties (Glass vs Plastic) using zero-shot vision-language matching.
- **Layer 3:** `Florence-2` provides dense context ("Crushed LDPE milk pouch").

### 2. Linguistic Override (Human-in-the-Loop)

Computer Vision can fail (e.g., Clear Plastic vs. Glass).

- **Whisper AI** listens to the user (English/Hindi/Regional).
- It **translates** intent and **overrides** the vision logic in real-time.
- *Example:* User says *"Ye kaanch hai"* -> System forces "Glass" tag even if it looks like plastic.

### 3. Vector Memory & Learning

The system gets smarter with use via the **Memory Agent**:

- **Episodic Memory:** Remembers specific objects user corrected previously.
- **Knowledge Base:** Stores material guidelines (e.g., "Shiny = MLP").
- **Retrieval:** Before finalizing a result, it queries **Qdrant** to see if it has seen this object before.

### 4. Semantic Market Matching

We replaced rigid SQL queries with **Vector Search**.

- Factory requirements (PDF Manifests) are parsed and embedded into **Qdrant**.
- Trash scans are matched based on **Semantic Proximity**.
- *Result:* "Dirty Polypropylene" automatically matches with "Industrial Polymer Recycler" buy orders.

### 5. Technical HUD & Agent Pipeline Visualization

The frontend provides a rich, wearable-style HUD. It renders a dedicated vertical timeline (**Agent Pipeline Flow**) that traces the AI's execution log in real-time: *Input Ingestion -> Bicubic Preprocess -> YOLOv11m Inference -> SigLIP Classification -> Contamination Analytics -> Deterministic Routing -> MQTT Dispatch.*

### 6. Built-in Pitch Deck

The application includes a hidden **Presentation Mode**. Swipe from the left edge of the screen to reveal the project slide deck dynamically rendered in React, fully customizable and pre-configured.

---

## 🛠️ Tech Stack

- **Orchestration:** FastAPI (Async Python)
- **Vector Database:** **Qdrant** (Local HNSW Index)
- **Vision:** Ultralytics YOLOv11m (fine-tuned on **TACO** dataset, 15 classes) + Microsoft Florence-2 + Google SigLIP
- **Audio:** OpenAI Whisper (Base) + FFmpeg
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
- **Frontend:** React + Vite + Tailwind CSS + Framer Motion

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- Node.js & npm
- **FFmpeg** (Required for Audio Processing)
  - *Linux:* `sudo apt install ffmpeg`
  - *Mac:* `brew install ffmpeg`
  - *Windows:* `winget install ffmpeg`

### 1. Backend Setup (Python)

```bash
# Clone the repository
git clone https://github.com/deepdas2607/reloop.git
cd reloop

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run the API Server
python main.py
```

### 2. Frontend Setup (React)

```bash
# Open a new terminal in the project root
cd frontend

# Install Node modules
npm install

# Run the UI
npm run dev
```

## 3. Usage Guide

- **Scan/Upload:** Point camera at waste. Click capture or Just Upload from Gallery.
- **Speak (Opt.)**: Click the mic to add context ("This is hazardous or This is plastic").
- **Market:** Click the 'Layout' icon to switch to B2B Market view.
- **Upload:** Use the "Import Manifest" button to upload a factory PDF requirement.
