from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

def normalize_text(text: str) -> str:
    prompt = f"Normalize this Albanian text into standard literary Albanian:\n{text}"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**inputs, max_length=256)

    return tokenizer.decode(output[0], skip_special_tokens=True)