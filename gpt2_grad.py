import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sipit_grad import sipit

model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hi, my name is Mikolaj Slowikowski."
inputs = tokenizer(text, return_tensors="pt").to(device)
orig_ids = inputs.input_ids[0].tolist()

print(f"\nOrginal: {text}")
print(f"Orginal ID: {orig_ids}")

with torch.no_grad():
    out = model(**inputs)
    target_hidden = out.hidden_states[-1][0]

print(target_hidden.shape)

rec_ids = sipit(model, target_hidden, tokenizer)

print(f"\nInverted ID: {rec_ids}")
print(f"Inverted text: {tokenizer.decode(rec_ids)}")