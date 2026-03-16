# Qwen-3 8B LoRA Fine-Tuned Recipe Dialogue Model

## Project Overview
This project fine-tunes **Qwen-3 8B** using LoRA on an open recipe dataset, enabling the model to understand and answer cooking-related questions.

The dataset contains about **13,000 recipes**, each including:
- Title
- Ingredients
- Instructions

The model can be used for:

- Recipe question answering  
- Generating cooking steps  
- Serving as a recipe agent for chat applications  

The fine-tuned model can also be deployed using **vLLM** for efficient and low-latency inference.

---

# Dataset

- **Source:** Open Recipe Dataset  
- Originally scraped from the **Epicurious website** and uploaded to Kaggle as  
  *Food Ingredients and Recipes Dataset*  

- **Processing:**
  - Removed images
  - Retained only text fields
  - Converted to JSON dialogue format for SFT training

### Example Data Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How do I make Miso-Butter Roast Chicken With Acorn Squash Panzanella?"
    },
    {
      "role": "assistant",
      "content": "Ingredients: ... Instructions: ..."
    }
  ]
}

### Using the Fine-Tuned Model

from transformers import AutoTokenizer, AutoModelForCausalLM
```python
tokenizer = AutoTokenizer.from_pretrained("models/qwen3-8b-lora")
model = AutoModelForCausalLM.from_pretrained("models/qwen3-8b-lora")

prompt = "How do I make Crispy Salt and Pepper Potatoes?"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=500
)

print(tokenizer.decode(outputs[0]))
