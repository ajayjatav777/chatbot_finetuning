import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "01-ai/Yi-34B-Chat"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load custom dataset from CSV file
df = pd.read_csv("/home/webtech/Desktop/hazoor/Chatbot/output.csv")

# Tokenize the dataset
tokenized_data = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

# Prepare custom dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=tokenized_data["input_ids"],
    block_size=128,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./custom_fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()