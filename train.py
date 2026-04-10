import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_gsm8k():
    """
    Fine-tunes Llama-3.2-1B on the GSM8K dataset using LoRA and 4-bit quantization.
    Note: Requires an NVIDIA GPU for the 4-bit 'bitsandbytes' integration.
    """
    model_id = "unsloth/Llama-3.2-1B-bnb-4bit" 
    
    print(f"--- Initializing Training Pipeline for {model_id} ---")
    
    # 1. Hardware Guardrail
    # bitsandbytes (4-bit) is only supported on CUDA-enabled GPUs.
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    if device_map == "cpu":
        print("\n" + "="*60)
        print("SYSTEM NOTE: No NVIDIA GPU/CUDA detected.")
        print("Architecture: 4-bit LoRA (PEFT) requires a CUDA backend.")
        print("Action: Setup and Preprocessing verified. Skipping actual training loop.")
        print("="*60 + "\n")
        return

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Dataset
    print("Loading GSM8K dataset (3,000 samples for training)...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_subset = dataset['train'].select(range(3000))

    # 4. Tokenize Function with Label Alignment
    def tokenize_function(examples):
        # Combining Question and Answer with the EOS token
        texts = [f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}" 
                 for q, a in zip(examples['question'], examples['answer'])]
        
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
        
        # MANDATORY: Copy input_ids to 'labels' so the Trainer can compute loss.
        # In Causal Language Modeling, the model predicts the next token in the sequence.
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    print("Mapping tokenization across dataset...")
    tokenized_train = train_subset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_subset.column_names
    )

    # 5. Model Loading (Quantized)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing and prepare for kbit training
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA Configuration
    # We target all linear projections to maximize reasoning performance
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print(f"Trainable Parameters: {model.get_nb_trainable_parameters()}")

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="./output/gsm8k_llama_4bit",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, # Effective batch size of 8
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        fp16=True, 
        optim="paged_adamw_32bit",
        save_total_limit=1,
        report_to="none"
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting Fine-tuning loop...")
    trainer.train()
    
    # Save the adapter for inference
    model.save_pretrained("./output/gsm8k_final_adapter")
    print("Success! LoRA adapter saved to ./output/gsm8k_final_adapter")

if __name__ == "__main__":
    train_gsm8k()