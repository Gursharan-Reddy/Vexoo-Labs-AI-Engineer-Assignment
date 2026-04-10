Vexoo Labs AI Assignment - B Gursharan Reddy

Ingestion & Pyramid:
Implemented a character-based 2-page sliding window with 50% overlap to maintain context. The Knowledge Pyramid uses four layers (Raw, Summary, Theme, Distilled) stored in a flat list for fast TF-IDF similarity retrieval. This ensures that if the raw text is too dense, the system can still match via distilled keywords.

GSM8K Setup:

Model: Llama 3.2 1B (via Hugging Face).

Technique: PEFT/LoRA (Rank 8) to ensure fine-tuning is possible on consumer GPUs.

Data: Split 3000/1000. Optimized with gradient accumulation.

Design Decisions:

Chose TF-IDF/Cosine Similarity for retrieval to avoid the overhead of a vector database for a small assignment.

Modularized the code into ingestion.py and train.py for "Plug-and-Play" capability.