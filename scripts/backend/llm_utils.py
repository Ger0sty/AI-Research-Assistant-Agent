# backend/llm_utils.py
import json
import os
import re
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

def call_llm_json(prompt: str, max_new_tokens: int = 512, system: str = None) -> dict:
    """
    Calls a chat-formatted LLaMA model using apply_chat_template,
    guaranteeing that system and user messages are separated.
    Returns the FIRST valid JSON object found in the assistant reply.
    """
    try:
        tok, mdl = get_model_and_tokenizer()

        # Default system instruction if none provided
        if system is None:
            system = (
                "You are a scientific reasoning assistant. "
                "You must output ONLY valid JSON. "
                "Never include extra commentary, explanations, or examples."
            )

        # Chat-style message list
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        # Convert chat messages to model input
        inputs = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(mdl.device)

        with torch.inference_mode():
            out = mdl.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
            )

        text = tok.decode(out[0], skip_special_tokens=True)
        print("[DEBUG] Raw LLM chat output:", text, flush=True)

        # --- Extract FIRST JSON object only ---
        i = text.find("{")
        if i < 0:
            return {"error": "No JSON found", "raw_output": text}
        j = text.find("}", i)
        if j < 0:
            return {"error": "No closing brace", "raw_output": text}

        json_str = text[i:j+1]

        # Clean and parse
        json_str = json_str.replace("\n", " ").strip()
        data = json.loads(json_str)
        print("[DEBUG] Parsed JSON:", data, flush=True)
        return data

    except Exception as e:
        print("[call_llm_json] ERROR:", e, flush=True)
        return {"error": str(e)}

def call_llm_json_last(prompt: str, max_new_tokens: int = 512, system: str = None) -> dict:
    """
    Same as call_llm_json, but extracts the LAST JSON object from the LLM output.
    Required for paper explanations because the model prints example JSON first.
    """
    # First, get the raw text using your existing logic
    try:
        tok, mdl = get_model_and_tokenizer()

        if system is None:
            system = (
                "You are a scientific reasoning assistant. "
                "You must output ONLY valid JSON. "
                "Never include extra commentary or multiple JSON blocks."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        inputs = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(mdl.device)

        with torch.inference_mode():
            out = mdl.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
            )

        text = tok.decode(out[0], skip_special_tokens=True)
        print("[DEBUG] Raw LLM chat output:", text, flush=True)

        # Find ALL { ... } objects
        json_candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)

        if not json_candidates:
            return {"error": "No JSON object found", "raw_output": text}

        # Take the LAST JSON object
        json_str = json_candidates[-1].replace("\n", " ").strip()

        data = json.loads(json_str)
        print("[DEBUG] Parsed LAST JSON:", data, flush=True)
        return data

    except Exception as e:
        print("[call_llm_json_last] ERROR:", e, flush=True)
        return {"error": str(e)}