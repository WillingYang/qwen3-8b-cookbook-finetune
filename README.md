# Qwen-3 8B LoRA Fine-Tuned Recipe Dialogue Model

## Project Overview
This project fine-tunes **Qwen-3 8B** using LoRA on an open recipe dataset, enabling the model to understand and answer cooking-related questions. The dataset contains about 13,000 recipes, each including a title, ingredients, and instructions.  

The model can be used for:
- Recipe question answering  
- Generating cooking steps  
- Serving as a recipe agent for chat applications  

The fine-tuned model can be deployed using **vLLM** for low-latency inference in recipe agent scenarios.

---

## Dataset
- Source: Open Recipe Dataset (originally scraped from the Epicurious website and uploaded to Kaggle as [Food Ingredients and Recipes Dataset](https://www.kaggle.com/datasets/<dataset-identifier>))  
- Data Cleaning: Images were removed; only text content (Title, Ingredients, Instructions) is retained.  
- Format: JSON dialogue format. Each record contains user input and assistant output, e.g.:

```json
{
  "messages": [
    {"role": "user", "content": "How do I make Miso-Butter Roast Chicken With Acorn Squash Panzanella?"},
    {"role": "assistant", "content": "Ingredients: ... Instructions: ..."}
  ]
}

## LoRA Fine-Tuning

### Data Preparation
- JSON dataset: `data/recipes_sft.jsonl`
- Entire dataset is used for training (**no validation split**)

## Using the Fine-Tuned Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("models/qwen3-8b-lora")
model = AutoModelForCausalLM.from_pretrained("models/qwen3-8b-lora")

# Example prompt
prompt = "How do I make Crispy Salt and Pepper Potatoes?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
