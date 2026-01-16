import os
import json
import time
from typing import Any, Dict, List
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np

from openai import OpenAI

from .utils import (
    normalize_money,
    build_llm_input_text,
    pair_items_from_ocr_lines,
    postprocess,
    dedup_candidates,
)

# ========== Config ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK_API_KEY is empty. Set it before starting.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# OCR language: for Germany receipts use "german"; for mixed try "en"
OCR_LANG = os.getenv("OCR_LANG", "german")

# --------- OCR lazy init (important for Render stability) ----------
_ocr = None
_ocr_lock = Lock()

def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                print(f"[OCR] initializing PaddleOCR lang={OCR_LANG} ...")
                _ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG)
                print("[OCR] PaddleOCR ready.")
    return _ocr
# ------------------------------------------------------------------

app = FastAPI(title="Receipt AI (OCR + DeepSeek)", version="1.2.0")

# Paths
APP_DIR = Path(__file__).resolve().parent          # .../receipt_ai/app
BASE_DIR = APP_DIR.parent                          # .../receipt_ai
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Serve static files once
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"Missing {INDEX_HTML}")
    return FileResponse(str(INDEX_HTML))


def run_ocr_bytes(image_bytes: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    np_img = np.array(img)

    ocr = get_ocr()
    result = ocr.ocr(np_img, cls=True)

    lines: List[Dict[str, Any]] = []
    full: List[str] = []

    for block in result:
        for entry in block:
            text = normalize_money((entry[1][0] or "").strip())
            conf = float(entry[1][1])
            if text:
                lines.append({"text": text, "confidence": conf})
                full.append(text)

    return {"ocr_text": "\n".join(full), "ocr_lines": lines}


def build_messages(ocr_text: str, currency: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = f"""
You are a receipt understanding engine.
Output ONLY valid JSON (no markdown, no commentary).

You will receive:
1) OCR text lines (may contain errors).
2) Item candidates (paired locally as name + line_total, plus optional weight hints).

Tasks:
- Fix obvious OCR errors (e.g., 0->O, 1->l, ALDT->ALDI).
- Extract: merchant, datetime, items, totals.
- For items, use candidates primarily; use OCR text as backup.
- Classify each item.

Return JSON schema:
{{
  "merchant": string|null,
  "datetime": string|null,
  "currency": "{currency}",
  "items": [
    {{
      "name_raw": string,
      "name": string,
      "qty": number|null,
      "unit_price": number|null,
      "line_total": number|null,
      "category": string,
      "confidence": number
    }}
  ],
  "totals": {{
    "subtotal": number|null,
    "tax": number|null,
    "total": number|null
  }}
}}

Categories (choose exactly one):
- 生鲜蔬果
- 肉禽海鲜
- 乳制品/蛋奶
- 零食/坚果
- 饮料
- 主食/调味/粮油
- 日用品/家居
- 其他

Examples:
- onions, potatoes, tomatoes, grapes, strawberries, carrots, salad, zucchini, garlic -> 生鲜蔬果
- turkey, chicken, beef, pork -> 肉禽海鲜
- cheese, sour cream, milk, yogurt, eggs -> 乳制品/蛋奶
- nuts, bars, chips, cookies -> 零食/坚果
- cola, juice, water -> 饮料
- rice, pasta, sauce, chili/sriracha, oil -> 主食/调味/粮油
- sport cap, detergent, tissue -> 日用品/家居

Rules:
- Keep duplicate lines as separate items (do NOT merge).
- Parse amounts carefully; decimal separator is '.'.
- If datetime appears glued (e.g., "09/19/2115:03"), split into date and time.
- confidence is 0~1.
""".strip()

    user = {
        "ocr_text": ocr_text,
        "item_candidates": candidates,
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def call_deepseek(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    last_err = None
    for attempt in range(2):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=60,
            )
            content = resp.choices[0].message.content or ""
            dt = time.time() - t0
            print(f"[DeepSeek] ok attempt={attempt+1} in {dt:.2f}s, chars={len(content)}")
            return json.loads(content)
        except Exception as e:
            last_err = e
            print(f"[DeepSeek] failed attempt={attempt+1}: {repr(e)}")
            time.sleep(0.5)

    raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(last_err)}")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/receipt/parse")
async def parse_receipt(
    file: UploadFile = File(...),
    currency: str = Query(default="EUR"),
    debug: bool = Query(default=False),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp supported.")

    image_bytes = await file.read()

    # 1) OCR
    ocr_out = run_ocr_bytes(image_bytes)
    if not ocr_out["ocr_text"].strip():
        raise HTTPException(status_code=422, detail="OCR produced empty text. Try a clearer photo.")

    # 2) Filter OCR text (reduce noise/tokens)
    filtered_text = build_llm_input_text(ocr_out["ocr_lines"])

    # 3) Local pairing (reduce LLM mistakes + fewer tokens)
    candidates = pair_items_from_ocr_lines(ocr_out["ocr_lines"])
    candidates = dedup_candidates(candidates)

    # 4) LLM
    messages = build_messages(filtered_text, currency=currency, candidates=candidates)
    structured = call_deepseek(messages)

    # 5) Postprocess (strip whitespace, clamp confidence, normalize)
    structured = postprocess(structured)

    # attach debug info only if requested
    if debug:
        structured["raw"] = ocr_out
        structured["debug"] = {
            "filtered_text": filtered_text,
            "item_candidates": candidates,
            "model": DEEPSEEK_MODEL,
            "ocr_lang": OCR_LANG,
        }

    return JSONResponse(content=structured)
