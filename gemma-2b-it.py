from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")




while True:
    print("-----------------------------------------------------")
    input_text = (input("Prompt: "))
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_length=250, eos_token_id= tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0]))
    print("-----------------------------------------------------")