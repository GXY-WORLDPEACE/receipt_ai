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

OCR_LANG = os.getenv("OCR_LANG", "german")

# 限制是否串行处理（Render 小机器上更稳；如果你升配可以关掉）
SERIALIZE_REQUESTS = os.getenv("SERIALIZE_REQUESTS", "1") == "1"

if not DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK_API_KEY is empty. Set it before starting.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# --------- OCR lazy init (important for Render stability) ----------
_ocr = None
_ocr_lock = Lock()

def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                print(f"[OCR] initializing PaddleOCR lang={OCR_LANG} ...")
                # show_log=False：避免日志过多 + 更稳
                _ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, show_log=False)
                print("[OCR] PaddleOCR ready.")
    return _ocr
# ------------------------------------------------------------------

# 可选：串行锁，避免并发时 OCR+LLM 抢 CPU/内存导致抖动
_req_lock = Lock()

app = FastAPI(title="Receipt AI (OCR + DeepSeek)", version="1.3.0")

# ---------- Static page ----------
APP_DIR = Path(__file__).resolve().parent          # .../receipt_ai/app
BASE_DIR = APP_DIR.parent                          # .../receipt_ai
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"Missing {INDEX_HTML}")
    return FileResponse(str(INDEX_HTML))


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/warmup")
def warmup():
    """
    部署后先访问一次 /warmup：
    - 触发 PaddleOCR 初始化（如果是懒加载）
    - 让 Render 的实例“热起来”
    """
    _ = get_ocr()
    return {"ok": True, "ocr_lang": OCR_LANG, "model": DEEPSEEK_MODEL}


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


def build_messages(
    ocr_text: str,
    currency: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
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
    """
    timeout + retry + clear logs.
    """
    last_err: Exception | None = None
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
            print(f"[LLM] ok model={DEEPSEEK_MODEL} attempt={attempt+1} {dt:.2f}s chars={len(content)}")

            try:
                return json.loads(content)
            except json.JSONDecodeError as je:
                print("[LLM] JSON decode error:", str(je))
                print("[LLM] raw content head:", content[:300])
                raise HTTPException(status_code=502, detail="LLM returned non-JSON output.")
        except HTTPException:
            raise
        except Exception as e:
            last_err = e
            print(f"[LLM] failed attempt={attempt+1}: {repr(e)}")
            time.sleep(0.6)

    raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(last_err)}")


@app.post("/v1/receipt/parse")
async def parse_receipt(
    file: UploadFile = File(...),
    currency: str = Query(default="EUR"),
    debug: bool = Query(default=False),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp supported.")

    # 可选：串行（更稳）
    lock = _req_lock if SERIALIZE_REQUESTS else None
    if lock:
        lock.acquire()

    try:
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

        # 5) Postprocess
        structured = postprocess(structured)

        if debug:
            structured["raw"] = ocr_out
            structured["debug"] = {
                "filtered_text": filtered_text,
                "item_candidates": candidates,
                "model": DEEPSEEK_MODEL,
                "ocr_lang": OCR_LANG,
                "serialize_requests": SERIALIZE_REQUESTS,
            }

        return JSONResponse(content=structured)

    finally:
        if lock:
            lock.release()
