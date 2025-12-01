
# 1️ Ensure GPU is available

import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Type:", torch.cuda.get_device_name(0))
else:
    print("⚠ No GPU detected. Enable GPU in Colab: Runtime > Change runtime type > GPU")
# 3️ Upload your 5 TSV files
from google.colab import files
uploaded = files.upload()

tsv_files = list(uploaded.keys())
print("Uploaded TSV files:", tsv_files)


# 4️ Convert TSV files → ONE training JSONL
import pandas as pd
import json

output_jsonl = "train_dataset.jsonl"

def build_prompt(row):
    """Creates a natural language instruction depending on the task type."""
    row = {k: v for k, v in row.items() if str(v) != "nan"}

    # Task A1 — Word Inclusion
    if "word1" in row and "word2" in row:
        return f"Write a funny joke that includes BOTH words: {row['word1']} and {row['word2']}.\nJoke:"

    # Task A2 — Headline Based Humor
    if "headline" in row:
        return f"Write a funny joke inspired by this news headline:\n\"{row['headline']}\"\nJoke:"

    # Task B2 — GIF Caption + Prompt
    if "prompt" in row:
        return f"Complete this humorous prompt:\n{row['prompt']}\nJoke:"

    # Task B1 — GIF-only caption
    if "url" in row:
        return f"Write a funny caption for this GIF:\n{row['url']}\nJoke:"

    # Default
    return "Write a humorous joke.\nJoke:"


with open(output_jsonl, "w", encoding="utf-8") as out:
    for f in tsv_files:
        df = pd.read_csv(f, sep="\t")

        for _, row in df.iterrows():
            prompt = build_prompt(row)

            # TEMP LABEL: we will fine-tune on self-generated jokes later
            target = "A funny joke goes here."

            out.write(json.dumps({
                "input_text": prompt,
                "target_text": target
            }, ensure_ascii=False) + "\n")

print("Saved training JSONL:", output_jsonl)


# 5️ Load training dataset
from datasets import Dataset

dataset = {
    "input_text": [],
    "target_text": []
}

with open(output_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        dataset["input_text"].append(obj["input_text"])
        dataset["target_text"].append(obj["target_text"])

dataset = Dataset.from_dict(dataset)
print(dataset)


# 6️ Load GPT-2 model (open-source, runs offline)
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.cuda()


# 7️ Tokenization
def tokenize(batch):
    inputs = tokenizer(
        batch["input_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    labels = tokenizer(
        batch["target_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_ds = dataset.map(tokenize, batched=True)


# 8️ Apply LoRA (fast, GPU-friendly fine-tuning)
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 9️ Training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./joke_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=25,
    save_steps=300,
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=collator,
    tokenizer=tokenizer,
)

trainer.train()

#  Save the trained model
model.save_pretrained("joke_model")
tokenizer.save_pretrained("joke_model")

print("🥳 Training Complete! Model saved.")


# Generate jokes!
def generate_joke(input_prompt, max_len=60):
    encoded = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **encoded,
        max_length=max_len,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("Joke:")[-1].strip()


# Example
print(generate_joke("Write a joke including BOTH words: banana and laptop.\nJoke:"))
