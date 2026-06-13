import os
import json
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/Danielfevargas16/travel-planner-llama"

def generate_hf(destino, presupuesto, viajero, ritmo, intereses):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN no configurado")

    prompt = f"Genera un itinerario para {destino}, presupuesto {presupuesto}, viajero {viajero}, ritmo {ritmo}, intereses: {intereses}"

    # Formato chat para Llama Instruct
    input_text = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_new_tokens": 2048,
            "do_sample": False,
            "return_full_text": False,
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]

    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end]

    return json.loads(json_str)
