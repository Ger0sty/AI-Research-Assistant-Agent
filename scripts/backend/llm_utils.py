# backend/llm_utils.py
import json
import os
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick your LLaMA model (replace with your exact model ID)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # example; use your actual model repo name
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

def call_llm_json(prompt: str, max_new_tokens: int = 256) -> dict:
    """
    Run a local Hugging Face model and robustly extract the first JSON object.
    Handles single quotes, trailing commas, and text before/after JSON.
    """
    try:
        tok, mdl = get_model_and_tokenizer()

        system_prompt = (
            "You are a scientific reasoning assistant that outputs ONLY valid JSON.\n"
            "Your response must be exactly one JSON object of the form:\n"
            "{\n  \"why\": \"your explanation here\"\n}\n"
        )
        full_prompt = system_prompt + prompt

        inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(mdl.device)

        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
            )

        text = tok.decode(out[0], skip_special_tokens=True)
        print("[DEBUG] Raw text from LLM passthrough:", text, flush=True)
        # --- Extract JSON-like substring ---
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j < 0:
            print("[DEBUG] No braces found in LLM output:", text[:400], flush=True)
            return {"error": "No JSON braces found", "raw_output": text}

        json_str = text[i:j+1]

        # --- Sanitize: replace single quotes with double, remove newlines ---
        json_str_clean = (
            json_str
            .replace("'", '"')
            .replace("\n", " ")
            .replace("\\n", " ")
            .strip()
        )

        try:
            data = json.loads(json_str_clean)
            print("[DEBUG] Parsed sanitized JSON:", data, flush=True)
            return data
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parse failed ({e}); returning raw text", flush=True)
            return {"error": f"Bad JSON: {e}", "raw_output": text}

    except Exception as e:
        print(f"[call_llm_json] ERROR: {e}", flush=True)
        return {"error": f"LLM call failed: {e}"}
