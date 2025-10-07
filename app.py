import os, json, math, time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from openai import OpenAI

# ---------- LLM Setup ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# You can also use the newer Responses API (single-call, stateful).
# See NOTE near the bottom for a drop-in variant using Responses API.  # Docs: https://platform.openai.com/docs/api-reference/responses

LLM_MODEL = "gpt-4o-mini"  # fast + good for tool-calls; swap as you like

# ---------- Demo Catalog (tiny, but realistic shape) ----------
PHONES = [
    {
        "id": "p_iphone15",
        "title": "iPhone 15",
        "brand": "Apple",
        "price": 999,
        "specs": {"ram": 6, "storage": 128, "battery": 3349, "camera_mp": 48},
        "features": ["camera", "ios", "magSafe", "oled", "wireless_charging"],
        "summary": "Great camera, iOS, MagSafe accessories, solid battery life."
    },
    {
        "id": "p_pixel9",
        "title": "Google Pixel 9",
        "brand": "Google",
        "price": 899,
        "specs": {"ram": 12, "storage": 256, "battery": 4700, "camera_mp": 50},
        "features": ["camera", "android", "ai_features", "wireless_charging", "ip68"],
        "summary": "AI-first Android with excellent still photography."
    },
    {
        "id": "p_s24",
        "title": "Samsung Galaxy S24",
        "brand": "Samsung",
        "price": 849,
        "specs": {"ram": 8, "storage": 256, "battery": 4000, "camera_mp": 50},
        "features": ["android", "display", "wireless_charging", "dex"],
        "summary": "Bright display, balanced specs, Samsung ecosystem."
    },
    {
        "id": "p_oneplus12",
        "title": "OnePlus 12",
        "brand": "OnePlus",
        "price": 799,
        "specs": {"ram": 16, "storage": 256, "battery": 5400, "camera_mp": 50},
        "features": ["android", "fast_charge", "gaming", "ltpo_display"],
        "summary": "Big battery, very fast charging, great for gaming."
    },
]

ACCESSORIES = [
    {
        "id": "a_magsafe_charger",
        "title": "MagSafe Charger 20W",
        "kind": "charger",
        "price": 49,
        "tags": ["magsafe", "wireless"],
        "compatible": {"brand": ["Apple"]}
    },
    {
        "id": "a_usb_pd_65w",
        "title": "USB-C PD 65W Fast Charger",
        "kind": "charger",
        "price": 39,
        "tags": ["usb_c", "pd", "fast_charge"],
        "compatible": {"brand": ["Samsung", "Google", "OnePlus", "Apple"]}
    },
    {
        "id": "a_pixel_case",
        "title": "Pixel 9 Silicone Case",
        "kind": "case",
        "price": 29,
        "tags": ["case", "pixel9_fit"],
        "compatible": {"model_ids": ["p_pixel9"]}
    },
    {
        "id": "a_s24_case",
        "title": "Galaxy S24 Protective Case",
        "kind": "case",
        "price": 35,
        "tags": ["case", "s24_fit"],
        "compatible": {"model_ids": ["p_s24"]}
    },
    {
        "id": "a_op12_case",
        "title": "OnePlus 12 Bumper Case",
        "kind": "case",
        "price": 25,
        "tags": ["case", "oneplus12_fit"],
        "compatible": {"model_ids": ["p_oneplus12"]}
    },
    {
        "id": "a_airpods_pro",
        "title": "AirPods Pro (USB-C)",
        "kind": "earbuds",
        "price": 249,
        "tags": ["anc", "bt5"],
        "compatible": {"brand": ["Apple", "Samsung", "Google", "OnePlus"]}
    },
]

# ---------- Embeddings ----------
# We'll embed "search_text" for phones & accessories to do semantic search.
EMBED_MODEL = "text-embedding-3-small"  # cheap + solid multilingual  # Docs: https://platform.openai.com/docs/guides/embeddings/embedding-models
def embed_texts(texts: List[str]) -> np.ndarray:
    out = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.array(e.embedding, dtype=np.float32) for e in out.data]
    return np.vstack(vecs)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-6
    return float(np.dot(a, b) / denom)

def build_search_corpus():
    items = []
    for x in PHONES:
        items.append({
            "type": "phone",
            "id": x["id"],
            "price": x["price"],
            "brand": x["brand"],
            "title": x["title"],
            "payload": x,
            "search_text": f'{x["title"]} {x["brand"]} {x["summary"]} {",".join(x["features"])} {x["specs"]}'
        })
    for x in ACCESSORIES:
        items.append({
            "type": "accessory",
            "id": x["id"],
            "price": x["price"],
            "brand": None,  # brand may be broad for accessories
            "title": x["title"],
            "payload": x,
            "search_text": f'{x["title"]} {x["kind"]} {",".join(x["tags"])} compatible:{x.get("compatible","")}'
        })
    vecs = embed_texts([i["search_text"] for i in items])
    for i, v in enumerate(vecs):
        items[i]["vec"] = v
    return items

SEARCH_CORPUS = build_search_corpus()

# ---------- Domain Tools the LLM can call ----------
def tool_search_catalog(query: str,
                        category: Optional[str] = None,
                        max_price: Optional[float] = None,
                        min_camera_mp: Optional[int] = None,
                        prefers: Optional[List[str]] = None,
                        brand: Optional[str] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic + rule filters over our tiny catalog."""
    qvec = embed_texts([query])[0]
    scored = []
    for item in SEARCH_CORPUS:
        if category and item["type"] != category:
            continue
        if brand and item["payload"].get("brand") and item["payload"]["brand"].lower() != brand.lower():
            continue
        if max_price and item["price"] > max_price:
            continue
        if min_camera_mp and item["type"] == "phone":
            if item["payload"]["specs"]["camera_mp"] < min_camera_mp:
                continue
        if prefers and item["type"] == "phone":
            # soft preference boost if feature keyword appears
            pref_boost = sum(1 for p in prefers if p in item["payload"]["features"])
        else:
            pref_boost = 0
        sim = cosine_sim(qvec, item["vec"])
        score = sim + 0.05 * pref_boost - 0.0005 * item["price"]
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1]["payload"] for x in scored[:top_k]]

def tool_accessory_bundle(phone_id: str, budget: Optional[float] = None) -> List[Dict[str, Any]]:
    phone = next((p for p in PHONES if p["id"] == phone_id), None)
    if not phone:
        return []
    brand = phone["brand"]
    picks = []
    for acc in ACCESSORIES:
        ok_brand = (not acc["compatible"].get("brand")) or (brand in acc["compatible"]["brand"])
        ok_model = (not acc["compatible"].get("model_ids")) or (phone_id in acc["compatible"]["model_ids"])
        if ok_brand and ok_model:
            picks.append(acc)
    # simple budget trim
    picks.sort(key=lambda a: a["price"])
    if budget:
        out, total = [], 0
        for acc in picks:
            if total + acc["price"] <= budget:
                out.append(acc); total += acc["price"]
        return out
    return picks[:3]

# ---------- Tool schema for Chat Completions ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search phones or accessories with semantic & filter criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text need, e.g., 'great camera under $900'"},
                    "category": {"type": "string", "enum": ["phone", "accessory"], "description": "Restrict category"},
                    "max_price": {"type": "number"},
                    "min_camera_mp": {"type": "integer"},
                    "prefers": {"type": "array", "items": {"type": "string"}, "description": "Soft preferences like 'camera','gaming','battery'"},
                    "brand": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "accessory_bundle",
            "description": "Suggest compatible accessories for a chosen phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_id": {"type": "string"},
                    "budget": {"type": "number"}
                },
                "required": ["phone_id"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a retail shopping copilot for mobiles & accessories.
- Always clarify missing constraints briefly (budget, brand affinity, camera/battery/display priors).
- When the user intent is clear, call tools to search and then propose a small ranked list (2–4 items).
- Include a 1–2 line rationale and a JSON block with fields: recommendations[], bundle[], follow_up_question.
- Ensure accessories are actually compatible with the chosen phone, not generic guesses.
- If budget too low for any match, ask to relax constraints.
Return concise, demo-ready output."""

# ---------- Session store (very light) ----------
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    text: str
    recommendations: List[Dict[str, Any]]
    bundle: List[Dict[str, Any]]
    follow_up_question: Optional[str] = None

def llm_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """One tool-call loop using Chat Completions (stable for live demos)."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.3
    )
    msg = resp.choices[0].message

    # If the LLM asked to call a tool:
    if msg.tool_calls:
        tool_results_msgs = []
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if name == "search_catalog":
                result = tool_search_catalog(**args)
            elif name == "accessory_bundle":
                result = tool_accessory_bundle(**args)
            else:
                result = {"error": f"unknown tool {name}"}

            # Append the *assistant* tool call and the *tool* result
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [call]
            })
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

        messages.extend(tool_results_msgs)

        # Final turn after tools
        final = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2
        ).choices[0].message

        content = final.content or ""
        return {"content": content}

    # No tool used, just return content
    return {"content": msg.content or ""}

def extract_json(payload: str) -> Dict[str, Any]:
    """Best-effort JSON extraction designed for demo stability."""
    try:
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            return json.loads(payload[start:end+1])
    except Exception:
        pass
    return {"recommendations": [], "bundle": [], "follow_up_question": None}

app = FastAPI(title="Agentic Mobile Recommender")

@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    session = SESSIONS.setdefault(inp.session_id, [])
    session.append({"role": "user", "content": inp.message})
    # add system only at first message
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + session

    out = llm_chat(msgs)
    text = out["content"]

    # Parse final JSON block
    parsed = extract_json(text)
    resp = ChatOut(
        text=text,
        recommendations=parsed.get("recommendations", []),
        bundle=parsed.get("bundle", []),
        follow_up_question=parsed.get("follow_up_question")
    )
    # Save assistant turn
    session.append({"role": "assistant", "content": text})
    return resp

# ----------------- OPTIONAL: Responses API variant (commented) -----------------
# The Responses API can do the same with a slightly different pattern and
# built-in hosted tools if desired. Cookbook example shows the new output shape.  # https://cookbook.openai.com/examples/responses_api/responses_example
# For custom tools you would:
#  1) client.responses.create(..., input=[...], tools=[{"type":"function", ...}], tool_choice="auto")
#  2) If output contains tool calls, run your Python tool(s)
#  3) client.responses.submit_tool_outputs(response_id=resp.id, tool_outputs=[{"tool_call_id":..., "output": "..."}])
#  4) Read final message from the follow-up response
