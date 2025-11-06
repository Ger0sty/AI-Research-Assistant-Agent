# backend/llm_utils.py
import json
import os
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick your LLaMA model (replace with your exact model ID)
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # example; use your actual model repo name
# Load model and tokenizer once
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, dtype=dtype, low_cpu_mem_usage=True
    )
    mdl.to(device); mdl.eval()
    return tok, mdl

def call_llm_json(prompt: str, max_new_tokens: int = 64) -> dict:  # smaller output
    """
    Call a local Hugging Face LLaMA model with a prompt and parse JSON output.
    """
    try:
        tokenizer, model = get_model_and_tokenizer()
        full = ("You are a helpful assistant that must output ONLY valid JSON.\n\n"
                + prompt + "\n\nOutput JSON only:")
        inputs = tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=512,     # cap prompt length to reduce activations
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # keep small
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        i, j = text.find("{"), text.rfind("}")
        return json.loads(text[i:j+1]) if i >= 0 and j >= 0 else {"error":"No JSON", "raw_output":text}
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}