import json


with open("data/tagged_corpus.json", "r") as f:
    corpus = json.load(f)


with open("data/generator_training.txt", "w") as f:
    for passage in corpus:
        themes = passage.get("themes", [])
        text = passage.get("text", "")
        
        if not themes or not text:
            continue
        
        # Format: [theme1][theme2] text
        theme_tags = "".join(f"[{t}]" for t in sorted(themes))
        f.write(f"{theme_tags} {text}\n\n")


print("Saved to data/generator_training.txt")
