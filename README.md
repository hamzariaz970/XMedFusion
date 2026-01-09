# Medical Imaging Report Generation Web Application

## Project Overview

This repository contains a web application developed to demonstrate a research oriented pipeline for medical imaging analysis and radiology report generation. The project focuses on **transparent, interpretable, and clinically grounded AI**, combining modern computer vision, large language models, and knowledge graph based reasoning.

The current implementation emphasizes **system design, UI flow, and research architecture**. Core AI components such as preprocessing, report generation, knowledge graph construction, and image report alignment are **hardcoded placeholders** for demonstration and prototyping purposes.

---

## Objectives

* Demonstrate an end to end workflow for medical image based report generation
* Showcase an agentic AI architecture suitable for clinical decision support
* Emphasize traceability, interpretability, and hallucination reduction
* Provide a clean, navigable frontend that mirrors a real clinical research system

---

## Application Pages and Functionality

### 1. Landing Page

* Introduces the project motivation and goals
* Explains the overall workflow from image upload to report generation
* Provides navigation to all other pages

---

### 2. X ray Upload and Report Generation Page

* Allows users to upload chest X ray images
* Supports multiple image formats:

  * DICOM
  * JPEG
  * PNG
* Applies image preprocessing (currently hardcoded)
* Displays a "Generating Report" status
* Shows a generated radiology report (currently hardcoded)

This page simulates how an AI system would process medical images and produce clinically structured text.

---

### 3. Knowledge Graph Visualization Page

* Visualizes a medical knowledge graph
* Displays nodes representing:

  * Anatomical regions
  * Abnormal findings
  * Clinical concepts
* Displays edges representing relationships between concepts
* All nodes and links are currently hardcoded for demonstration

This page highlights how structured medical knowledge can be used to ground and explain AI generated reports.

---

### 4. Image Report Mapping Page

* Displays the uploaded X ray image
* Shows the generated report alongside the image
* Highlights specific report sentences
* Maps each highlighted sentence to corresponding image regions
* Sentence to region mappings are currently hardcoded

This page demonstrates the concept of spatial grounding and explainability in medical AI systems.

---

## Research Architecture

The project is conceptually designed as an **agentic AI framework** for radiology report generation:

* **Vision Agent**

  * Detects abnormalities and visual features from medical images

* **Retrieval Agent**

  * Performs case based reasoning using similar historical examples

* **Draft Agent**

  * Generates an initial radiology report using LLMs

* **Refiner Agent**

  * Improves clarity, structure, and medical coherence

* **Synthesis Agent**

  * Produces the final clinically formatted report

* **Clinical Knowledge Graph**

  * Stores structured findings, spatial metadata, and relationships

* **Evidence Gate**

  * Verifies every diagnostic claim against image derived facts
  * Reduces hallucinations and improves interpretability

The current frontend mirrors this architecture even though backend reasoning is mocked.

---

## Technologies and Methods

**Frontend and System Design**

* Modern web frontend framework
* Multi page navigation with clean UI
* Placeholder driven prototyping

**AI and Research Stack (Conceptual and Evaluated Separately)**

* PyTorch
* Transformers: T5, FLAN T5, BART
* CNNs and Vision Transformers:

  * ResNet 101
  * Swin Transformer
* CLIP for vision language alignment
* Knowledge Graphs for structured reasoning
* Multi Agent Systems
* Python
* Medical imaging and DICOM processing

---

## Evaluation Summary

The proposed system was evaluated on public medical imaging datasets and demonstrated:

* Improved factual consistency compared to end to end report generation models
* Better clarity and clinical structure in generated reports
* Enhanced explainability through evidence linked claims

These results support the feasibility of transparent and accountable AI for healthcare decision support.

---

## Development Setup

```sh
# Clone the repository
git clone <YOUR_GIT_URL>

# Navigate into the project directory
cd <YOUR_PROJECT_NAME>

# Install dependencies
npm install

# Start the development server
npm run dev
```

---

## Deployment

This frontend can be deployed using any static or client side hosting service such as:

* GitHub Pages
* Vercel
* Netlify

Backend AI components are not deployed in this version and are represented through hardcoded placeholders.

---

## Disclaimer

This project is a **research prototype and demonstration system**.

* It is not a medical device
* It is not intended for clinical diagnosis
* Outputs are simulated and should not be used for real patient care

---

## License

Specify the appropriate license or state educational and research use only.
