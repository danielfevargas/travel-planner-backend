from groq import Groq
import os
import json
import torch
from dotenv import load_dotenv
from app.rag import get_travel_context
from app.local_model import generate_local
from app.hf_model import generate_hf

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_itinerary(preferences: dict) -> dict:
    destino = preferences.get("destination", "")
    start_date = preferences.get("start_date", "")
    end_date = preferences.get("end_date", "")
    presupuesto = preferences.get("budget", "moderado")
    tipo_viajero = preferences.get("traveler_type", "solo")
    ritmo = preferences.get("pace", "equilibrado")
    intereses = preferences.get("interests", [])
    intereses_str = ', '.join(intereses)

    # 1. GPU disponible → modelo local
    if torch.cuda.is_available():
        try:
            resultado = generate_local(destino, presupuesto, tipo_viajero, ritmo, intereses_str)
            print(f"[LOCAL] Itinerario generado para {destino}")
            return resultado
        except Exception as e:
            print(f"[LOCAL FALLÓ] {e}")

    # 2. Sin GPU pero hay HF_TOKEN → HuggingFace Inference API
    if os.getenv("HF_TOKEN"):
        try:
            resultado = generate_hf(destino, presupuesto, tipo_viajero, ritmo, intereses_str)
            print(f"[HF API] Itinerario generado para {destino}")
            return resultado
        except Exception as e:
            print(f"[HF API FALLÓ] {e}")

    # 3. Fallback → Groq
    print(f"[GROQ] Generando itinerario para {destino}...")

    contexto = get_travel_context(presupuesto, ritmo, tipo_viajero)

    prompt = f"""
Eres un experto planificador de viajes con conocimiento profundo de destinos mundiales.

REGLAS DE PLANIFICACIÓN:
{contexto}

PREFERENCIAS DEL USUARIO:
- Destino: {destino}
- Fechas: {start_date} al {end_date}
- Nivel de presupuesto: {presupuesto}
- Tipo de viajero: {tipo_viajero}
- Ritmo del viaje: {ritmo}
- Intereses: {', '.join(intereses)}

Usa tu conocimiento del destino para generar un itinerario detallado, realista y personalizado.
Incluye lugares reales, restaurantes conocidos, tips locales auténticos y horarios lógicos.

Responde SOLO con este JSON exacto, sin texto adicional ni markdown:
{{
  "destination": "nombre del destino",
  "total_days": número de días,
  "budget_level": "{presupuesto}",
  "traveler_type": "{tipo_viajero}",
  "days": [
    {{
      "day": 1,
      "title": "Título creativo del día",
      "activities": [
        {{
          "time": "09:00",
          "name": "Nombre del lugar o actividad",
          "description": "Descripción detallada y útil",
          "type": "cultural/gastronomy/outdoor/shopping/nightlife",
          "tip": "Consejo local específico"
        }}
      ],
      "restaurants": ["Restaurante real 1", "Restaurante real 2"],
      "local_tips": ["Tip local útil 1", "Tip local útil 2"]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())