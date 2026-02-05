# XMedFusion: Agentic Radiology Report Generation

## Project Overview

This repository contains a web application developed to demonstrate a research-oriented pipeline for medical imaging analysis and radiology report generation. The project focuses on **transparent, interpretable, and clinically grounded AI**, combining modern computer vision, large language models, and knowledge graph-based reasoning.

The current implementation emphasizes **system design, UI flow, and research architecture**. Core AI components such as preprocessing, report generation, knowledge graph construction, and image-report alignment are **hardcoded placeholders** for demonstration and prototyping purposes.

---

## Objectives

* Demonstrate an end-to-end workflow for medical image-based report generation
* Showcase an agentic AI architecture suitable for clinical decision support
* Emphasize traceability, interpretability, and hallucination reduction
* Provide a clean, navigable frontend that mirrors a real clinical research system

---

## Application Pages and Functionality

### 1. Landing Page
* Introduces the project motivation and goals
* Explains the overall workflow from image upload to report generation
* Provides navigation to all other pages
![alt text](screenshots\landing_page.png)

### 2. X-ray Upload and Report Generation Page
* Allows users to upload chest X-ray images
* Supports multiple image formats:
    * DICOM
    * JPEG
    * PNG
* Applies image preprocessing
* Displays a "Generating Report" status
* Shows a generated radiology report
![alt text](screenshots\upload_xray_page.png)

This page simulates how an AI system would process medical images and produce clinically structured text.

### 3. Knowledge Graph Visualization Page
* Visualizes a medical knowledge graph
* Displays nodes representing:
    * Anatomical regions
    * Abnormal findings
    * Clinical concepts
* Displays edges representing relationships between concepts
* All nodes and links are currently hardcoded for demonstration
![alt text](screenshots\kg_page.png)

This page highlights how structured medical knowledge can be used to ground and explain AI-generated reports.

### 4. Image-Report Mapping Page
* Displays the uploaded X-ray image
* Shows the generated report alongside the image
* Highlights specific report sentences
* Maps each highlighted sentence to corresponding image regions
* Sentence-to-region mappings are currently hardcoded

This page demonstrates the concept of spatial grounding and explainability in medical AI systems.

---

## Research Architecture

The project is conceptually designed as an **agentic AI framework** for radiology report generation:

* **Vision Agent:** Detects abnormalities and visual features from medical images
* **Retrieval Agent:** Performs case-based reasoning using similar historical examples
* **Draft Agent:** Generates an initial radiology report using LLMs
* **Refiner Agent:** Improves clarity, structure, and medical coherence
* **Synthesis Agent:** Produces the final clinically formatted report
* **Clinical Knowledge Graph:** Stores structured findings, spatial metadata, and relationships
* **Evidence Gate:** Verifies every diagnostic claim against image-derived facts to reduce hallucinations and improve interpretability

The current frontend mirrors this architecture even though backend reasoning is mocked.

---
## Technologies and Methods

**Frontend and System Design**
* Modern web frontend framework
* Multi-page navigation with clean UI

**AI and Research Stack**
* PyTorch
* **Ollama (Local LLM Inference)**
* Transformers: CLIP, Vision Transformers
* Multi-Agent Systems
* Python
* Medical imaging and DICOM processing

---

## 🛠️ Prerequisites & Setup

Before running the application, you must set up the **Dataset**, **Model Weights**, and **Ollama Backend**.

### 1. Dataset Setup
1.  Download the **IU X-Ray Dataset** from Kaggle:
    * [https://www.kaggle.com/datasets/areebbinnadeem/iu-xray](https://www.kaggle.com/datasets/areebbinnadeem/iu-xray)
2.  Create a folder named `data` in the root directory of this project.
3.  Extract and place the dataset files inside the `data` folder.

### 2. Custom Model Weights Setup
1.  Download the required model weights from the following link:
    * [Google Drive - Model Weights](https://drive.google.com/drive/folders/1ofqBS3rrLMdovOXTcNDudWpSV7nSTvd1?usp=sharing)
2.  Create a folder named `model_weights` in the root directory.
3.  Inside `model_weights`, create the following three sub-folders:
    * `Draft_Agent`
    * `KG_Agent`
    * `Vision_Agent`
4.  Place the relevant downloaded weight files into their respective sub-folders.

### 3. Ollama & LLM Setup (CRITICAL)
This project uses **Ollama** to run the medical LLM locally. You must install it and pull the specific model before running the backend.

1.  **Download Ollama:**
    * Go to [ollama.com](https://ollama.com) and download the installer for your OS (Windows/Mac/Linux).
    * Install Ollama.

2.  **Start the Ollama Server:**
    * Open a terminal and type:
      ```sh
      ollama serve
      ```
    * *Keep this terminal window open in the background.*

3.  **Download the Medical Model:**
    * Open a **new** terminal window and run the exact command below to pull the required 4-billion parameter medical model:
      ```sh
      ollama pull MedAIBase/MedGemma1.5:4b
      ```
    * *Note: This download is approximately 2.6 GB.*

---

##  Development Setup (Run Instructions)

To run the full application, you will need **two separate terminal windows** open simultaneously: one for the Backend (Python) and one for the Frontend (Node.js).

### Terminal 1: Backend Server

```sh
# 1. Clone the repository (if not already done)
git clone <YOUR_GIT_URL>
cd <YOUR_PROJECT_NAME>

# 2. Navigate to backend folder
cd backend

# 3. Create conda environment
conda create --name xmedfusion_env python=3.10
conda activate xmedfusion_env

# 4. Install dependencies
# (Ensure you have PyTorch with CUDA support if you have an NVIDIA GPU)
pip install -r requirements.txt

# 5. Start the Backend API
python app.py
```


You should see "Application startup complete" and "All AI Agents Ready"

### Terminal 2: Frontend Setup

```sh
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```


## Evaluation Summary

The proposed system was evaluated on public medical imaging datasets and demonstrated:

* Improved factual consistency compared to end-to-end report generation models
* Better clarity and clinical structure in generated reports
* Enhanced explainability through evidence-linked claims

These results support the feasibility of transparent and accountable AI for healthcare decision support.

---

## Deployment

This frontend can be deployed using any static or client-side hosting service such as:

* GitHub Pages
* Vercel
* Netlify

Backend AI components are not deployed in this version and are represented through hardcoded placeholders.

---

## Disclaimer

This project is a **research prototype and demonstration system**.

* It is not a medical device.
* It is not intended for clinical diagnosis.
* Outputs are simulated and should not be used for real patient care.

---

## Team & Affiliation

**Final Year Project (FYP) Team:**

* **Hamza Riaz** (hriaz.bscs22seecs@seecs.edu.pk)
* **Arham Haroon** (aharoon.bscs22seecs@seecs.edu.pk)
* **Maha Baig** (mbaig.bscs22seecs@seecs.edu.pk)

**Affiliation:**
MachVis Lab, School of Electrical Engineering and Computer Science (SEECS)
National University of Sciences and Technology (NUST), Islamabad, Pakistan

---

## License

This project is intended for **Educational and Research Use Only**.

