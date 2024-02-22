import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

compute_dtype = torch.float32
cache_path    = ''
device        = 'cpu'
model_id      = "mobiuslabsgmbh/aanaphi2-v0.1"
model         = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, 
                                                                  cache_dir=cache_path,
                                                                  device_map=device)
tokenizer     = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)

instruction_template = "### Human: "
response_template    = "### Assistant: "
def prompt_format(prompt):
    out = instruction_template + prompt + '\n' + response_template
    return out
model.eval()

streamer = TextStreamer(tokenizer)

@torch.no_grad()
def generate(prompt, max_length=500):
    prompt_chat = prompt_format(prompt)
    inputs      = tokenizer(prompt_chat, return_tensors="pt", return_attention_mask=True).to('cpu')
    outputs     = model.generate(**inputs, max_length=max_length, streamer=streamer,  eos_token_id= tokenizer.eos_token_id) 
    text        = tokenizer.batch_decode(outputs[:,:-1])[0]
    
    return text

while True:
    print("-----------------------------------------------------")
    print(generate(input("prompt: ")))
    print("-----------------------------------------------------")