from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("text", data_files={"train": "data/generator_training.txt"})


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


training_args = TrainingArguments(
    output_dir="models/generator",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16, # 16 for ThinkPad, 2 for tower
    gradient_accumulation_steps=1, # 1 for Thinkpad, 4 for tower
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
    fp16=False,
    use_mps_device=False,  # True For M1 Mac, False for Snapdragon ThinkPad or AMD/Nvidia Tower
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


trainer.train()
# trainer.train(resume_from_checkpoint=True)


model.save_pretrained("models/generator")
tokenizer.save_pretrained("models/generator")


print("Saved to models/generator/")
