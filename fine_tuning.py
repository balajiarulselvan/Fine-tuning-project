# -*- coding: utf-8 -*-
"""Fine-tuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gh4-C1277sY2eeVdc6XGlEsK5zy65Usv
"""

# Step 1: Install required packages
!pip install -qU transformers datasets peft bitsandbytes wandb

# Step 2: Import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Step 3: Load dataset
# Using databricks/dolly-15k as example dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Step 4: Create train, validation, test splits
small_dataset = dataset["train"].select(range(500))  # small subset for demo
train_test = small_dataset.train_test_split(test_size=0.2, seed=42)
val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = train_test["train"]
val_dataset = val_test["train"]
test_dataset = val_test["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Step 5: Format dataset examples
def format_example(example):
    return {"text": f"Instruction: {example['instruction']}\nResponse: {example['response']}"}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)
test_dataset = test_dataset.map(format_example)

# Step 6: Load tokenizer and set pad token
model_name = "distilgpt2"  # small model for demo

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # fix for models without pad token

# Step 7: Tokenize datasets
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # for causal LM
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Step 8: Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # increase for better results
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="none"  # disable wandb for simplicity
)

# Step 10: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Step 11: Train model
train_output = trainer.train()
print(train_output)

# Step 12: Evaluate model
eval_results = trainer.evaluate()
print(eval_results)

# Step 13: Inference example
prompt = "Instruction: What are the symptoms of diabetes?\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Generated response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Step 14: Save model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

!pip install --upgrade transformers

!pip install --upgrade transformers datasets

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",  # Works in transformers >=4.0.0
    save_strategy="no",
    report_to="none"
)

!pip show transformers
# Should show "Version: 4.51.3" or newer

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",  # ✅ Correct parameter name
    save_strategy="no",
    report_to="none"
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="epoch",  # ✅ Correct parameter name
    save_strategy="no",
    report_to="none"
)

!pip install --upgrade transformers

prompt = "Instruction: Diabetes symptoms?\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="epoch",  # ✅ Correct parameter name
    save_strategy="no",
    report_to="none"
)

!pip install --upgrade transformers datasets

from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load dataset and model
dataset = load_dataset("databricks/databricks-dolly-15k")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Define training arguments with eval_strategy
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="epoch",  # Works in v4.51.3
    save_strategy="no",
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
      # <-- Add this!
)
# Small subset for testing
trainer.train()

pip install --upgrade transformers>=4.39.0

!pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
!pip install transformers datasets

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Configure for batch processing
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # If not already set
model.to(device)
model.eval()

# Prepare questions

    healthcare_questions = [
    "What are the most common symptoms of diabetes?",
    "How does hypertension affect the body?",
    "What is the difference between a nurse practitioner and a physician assistant?",
    "Explain the importance of patient confidentiality in healthcare.",
    "What steps should you take if a patient is having a heart attack?",
    "How do vaccines work to protect against diseases?",
    "What are the main causes of kidney failure?",
    "Describe the process of taking a patient’s medical history.",
    "What is informed consent and why is it important?",
    "How do you handle a situation where a patient refuses treatment?",
    "What is the role of a physical therapist in patient recovery?",
    "Explain the difference between acute and chronic illnesses.",
    "What precautions should be taken to prevent hospital-acquired infections?",
    "How do you ensure effective communication with patients from diverse backgrounds?",
    "What are the ethical considerations in end-of-life care?"

]

# Batch processing
input_texts = ["Healthcare_question: " + q for q in Healthcare_questions]
inputs = tokenizer(input_texts, padding=True, return_tensors="pt", return_attention_mask=True).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.7,
        repetition_penalty=1.2
    )

# Extract answers (removing input questions)
input_lengths = inputs.attention_mask.sum(dim=1)
answers = [
    tokenizer.decode(outputs[i, input_lengths[i]:], skip_special_tokens=True)
    for i in range(len(Healthcare_questions))
]

# Print formatted results
for question, answer in zip(Healthcare_questions, answers):
    print(f"🧠 Q: {question}\n✅ A: {answer}\n")

# Save model (if not already done)
model.save_pretrained("/content/fine_tuned_model")
tokenizer.save_pretrained("/content/fine_tuned_model")

# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/content/fine_tuned_model",
    local_files_only=True  # Force local load
)
tokenizer = AutoTokenizer.from_pretrained(
    "/content/fine_tuned_model",
    local_files_only=True
)

from evaluate import load
squad_metric = load("squad")
predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
references = [{'answers': {'answer_start': [0], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
results = squad_metric.compute(predictions=predictions, references=references)
print(results)  # {'exact_match': 100.0, 'f1': 100.0}

# Commented out IPython magic to ensure Python compatibility.
# %pip install evaluate

from evaluate import load
squad_metric = load("squad")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/content/fine_tuned_model"  # Update this if your model is elsewhere

model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

while True:
    user_prompt = input("Ask a question (or type 'exit' to quit): ")
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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    print(f"Model: {response}\n")
