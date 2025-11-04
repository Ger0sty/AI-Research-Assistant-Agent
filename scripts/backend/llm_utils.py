# backend/llm_utils.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick your LLaMA model (replace with your exact model ID)
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # example; use your actual model repo name

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

def call_llm_json(prompt: str, max_new_tokens: int = 512) -> dict:
    """
    Call a local Hugging Face LLaMA model with a prompt and parse JSON output.
    """
    # Add generation instructions
    full_prompt = (
        "You are a helpful assistant that must output ONLY valid JSON.\n\n"
        + prompt
        + "\n\nOutput JSON only:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract and parse JSON
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_output": text}
