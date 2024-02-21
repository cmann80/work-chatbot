
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer , AutoModelForCausalLM
import torch 

tokenizer = AutoTokenizer.from_pretrained("mobiuslabsgmbh/aanaphi2-v0.1")

base_model = AutoModelForCausalLM.from_pretrained(
    "mobiuslabsgmbh/aanaphi2-v0.1",
    torch_dtype=torch.float32,
    device_map='cpu',
    )

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)
local_llm = HuggingFacePipeline(pipeline=pipe)
pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

template = """
### Instruction:
{instruction}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

while True:
    print("-----------------------------------------------------")
    print(llm_chain.invoke(input("Prompt: "))["text"])
    print("-----------------------------------------------------")