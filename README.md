# Vexoo Labs AI Engineer Assignment
**Candidate:** B Gursharan Reddy  
**Project:** Document Ingestion (RAG) & Reasoning Model Fine-tuning

## Project Overview
This repository contains a modular solution for the Vexoo Labs assignment, featuring:
1. **Document Ingestion:** A 2-page sliding window strategy with a 4-layer Knowledge Pyramid.
2. **Fine-tuning:** A LoRA-based SFT pipeline for GSM8K reasoning using Llama-3.2-1B.
3. **Reasoning-Aware Adapter:** A dynamic routing component that activates specialized logic based on query intent.

## File Structure
- `main.py`: Entry point to demonstrate all assignment parts.
- `ingestion.py`: Logic for sliding windows and the hierarchical Knowledge Pyramid.
- `train.py`: Fine-tuning configuration for Llama-3.2-1B (Optimized for 4-bit LoRA).
- `bonus.py`: The reasoning-aware routing adapter implementation.
- `requirements.txt`: Python dependencies.

## Usage
1. Install dependencies:
   pip install -r requirements.txt

2. Execute the demonstration:

python main.py

---

#### 2. One-Page Summary (PDF)
> **Vexoo Labs AI Assignment Summary**
> **Candidate:** B Gursharan Reddy
>
> **Ingestion & Knowledge Pyramid**
> - **Strategy:** Implemented a character-based sliding window (2000 chars) with a 1000-character overlap to preserve semantic context across chunks.
> - **Pyramid Structure:** Developed a 4-layer abstraction (Raw, Summary, Theme, Distilled) to allow the system to match queries against multiple levels of granularity.
> - **Retrieval:** Leverages TF-IDF and Cosine Similarity to select the most relevant pyramid level, ensuring robust retrieval even when exact phrasing differs.
>
> **GSM8K Fine-tuning Setup**
> - **Model:** Llama-3.2-1B (Quantized 4-bit) for efficient local training.
> - **Technique:** LoRA (Rank 16, Alpha 32) targeting all linear projections to capture complex reasoning patterns with minimal trainable parameters (0.27% of total parameters).
> - **Optimization:** Utilizes Paged AdamW and Gradient Accumulation (8 steps) to maximize stability on consumer hardware.
>
> **Design Decisions & Assumptions**
> - Used 4-bit quantization to reduce memory footprint and bandwidth requirements.
> - Modularized the ingestion and training pipelines to ensure "Plug-and-Play" extensibility.
> - Implemented a routing-based reasoning adapter to handle domain-specific (Math, Legal) vs. general queries efficiently.

---
