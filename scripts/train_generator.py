from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


model_name = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("text", data_files={"train": "data/generator_training.txt"})


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


training_args = TrainingArguments(
    output_dir="models/generator",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
    fp16=False,
    use_mps_device=True,  # True For M1 Mac False for Snapdragon ThinkPad
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


trainer.train()


model.save_pretrained("models/generator")
tokenizer.save_pretrained("models/generator")


print("Saved to models/generator/")
