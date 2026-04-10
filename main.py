import os
from ingestion import KnowledgePyramid
from train import train_gsm8k
from bonus import ReasoningRouter

def main():
    print("--- Vexoo Labs AI Engineer Assignment ---")
    
    # 1. Demonstrate Part 1: Ingestion & Knowledge Pyramid
    print("\n[Step 1] Initializing Knowledge Pyramid...")
    sample_text = (
        "The transformer architecture, introduced in 2017, revolutionized NLP. "
        "It uses self-attention mechanisms to weigh the significance of different parts of input data. "
        "Large Language Models like Llama 3.2 utilize this architecture for reasoning tasks."
    ) * 20  # Artificial bulk for sliding window
    
    pyramid = KnowledgePyramid(sample_text)
    query = "What architecture do LLMs use?"
    result = pyramid.query(query)
    print(f"Query: {query}")
    print(f"Retrieved Context: {result[:150]}...")

    # 2. Demonstrate Bonus: Reasoning Router
    print("\n[Step 2] Testing Reasoning-Aware Adapter...")
    router = ReasoningRouter()
    queries = ["What is 15 * 12?", "Review this legal contract.", "Tell me a joke."]
    for q in queries:
        print(f"Input: {q} -> {router.handle(q)}")

    # 3. Part 2: Model Training (Optional Trigger)
    choice = input("\n[Step 3] Do you want to start GSM8K fine-tuning? (y/n): ")
    if choice.lower() == 'y':
        print("Starting training script... (This requires a GPU)")
        train_gsm8k()
    else:
        print("Skipping training. Setup is verified.")

if __name__ == "__main__":
    main()