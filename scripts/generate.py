from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Load fine-tuned model ---
tokenizer = AutoTokenizer.from_pretrained("models/generator")
model = AutoModelForCausalLM.from_pretrained("models/generator")

def generate(themes, max_length=200):
    """Generate Gothic text conditioned on themes."""
    theme_tags = "".join(f"[{t}]" for t in sorted(themes))
    
    inputs = tokenizer(theme_tags, return_tensors="pt")
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# --- Test ---
print("=== supernatural + setting ===")
print(generate(["supernatural", "setting"]))

print("\n=== love + sorrow ===")
print(generate(["love", "sorrow"]))

print("\n=== harm + villainy + anguish ===")
print(generate(["harm", "villainy", "anguish"]))