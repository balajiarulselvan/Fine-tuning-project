# Fine-tuning-project


# Healthcare LLM Fine-Tuning Demo

This project demonstrates how to fine-tune a language model (LLM) for healthcare question answering using Hugging Face Transformers and the Dolly-15k dataset. It includes all steps from data preparation to interactive inference.

## Features

- Fine-tunes a base LLM (e.g., DistilGPT-2) on healthcare-related instructions and responses
- Supports evaluation and error analysis
- Provides an interactive prompt interface for testing model outputs
- Includes batch inference for a list of healthcare questions

## Setup

### 1. Clone the Repository

## Usage

### 1. Fine-Tune the Model

Run the notebook or script to fine-tune the model on the healthcare subset of the Dolly-15k dataset. Example code is included in `notebook.ipynb` or `finetune.py`.

### 2. Save and Load the Model

After training, save your model:
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

text

Load for inference:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("fine_tuned_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model", local_files_only=True)

text

### 3. Interactive Prompting

Test your model with custom healthcare questions:
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

while True:
user_prompt = input("Ask a healthcare question (or type 'exit' to quit): ")
if user_prompt.lower() == "exit":
break
prompt = f"Instruction: {user_prompt}\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
outputs = model.generate(
**inputs,
max_new_tokens=50,
do_sample=True,
temperature=0.8,
top_p=0.9,
repetition_penalty=1.1
)
response = tokenizer.decode(outputs, skip_special_tokens=True)
if "Response:" in response:
response = response.split("Response:")[-1].strip()
print(f"Model: {response}\n")

text

### 4. Batch Inference Example

healthcare_questions = [
"What are the most common symptoms of diabetes?",
"How does hypertension affect the body?",
# ... add more questions as needed
]

for question in healthcare_questions:
prompt = f"Instruction: {question}\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
outputs = model.generate(
**inputs,
max_new_tokens=50,
do_sample=True,
temperature=0.8,
top_p=0.9,
repetition_penalty=1.1
)
response = tokenizer.decode(outputs, skip_special_tokens=True)
if "Response:" in response:
response = response.split("Response:")[-1].strip()
print(f"Q: {question}\nA: {response}\n")


## Example Healthcare Questions

See [`healthcare_questions.py`](healthcare_questions.py) for a sample list.

## Evaluation

Evaluate your model using metrics like ROUGE or accuracy:
import evaluate
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=pred_list, references=ref_list)
print(results)
