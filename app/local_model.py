import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer

    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    finetuned_path = "./travel-model-final"

    _tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    _model = PeftModel.from_pretrained(base_model, finetuned_path)
    _model.eval()
    print("Modelo local cargado")

def get_model():
    if _model is None:
        _load_model()
    return _model, _tokenizer

def generate_local(destino, presupuesto, viajero, ritmo, intereses):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch no está instalado")

    model, tokenizer = get_model()

    prompt = f"Genera un itinerario para {destino}, presupuesto {presupuesto}, viajero {viajero}, ritmo {ritmo}, intereses: {intereses}"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    start = response.find("{")
    end = response.rfind("}") + 1
    json_str = response[start:end]

    return json.loads(json_str)
