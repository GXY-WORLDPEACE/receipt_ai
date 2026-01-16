import re
from typing import Any, Dict, List, Optional

# ---- Money normalization ----
def normalize_money(text: str) -> str:
    # "2,50" -> "2.50"
    return re.sub(r"(\d),(\d)", r"\1.\2", text)

# ---- Line filters ----
_BAD_PREFIX = (
    "www.", "http", "store #", "store#", "your cashier", "cashier",
    "ref/seq", "auth", "aid", "tvr", "iad", "tsi", "arc", "entrymode",
    "++approved", "approved", "pin", "visa", "mastercard", "amex",
)

def is_noise_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    tl = t.lower()
    if "xxxxxxxx" in tl:
        return True
    for p in _BAD_PREFIX:
        if tl.startswith(p):
            return True
    # very long payment blobs
    if len(t) > 80 and any(x in tl for x in ("aida", "tvr", "iad", "tsi")):
        return True
    return False

# ---- Price patterns ----
# Examples: "3.99 FA", "2.99FB", "66.66"
PRICE_RE = re.compile(r"(?<!\d)(\d+\.\d{2})(?!\d)")
WEIGHT_RE = re.compile(r"(?<!\d)(\d+(\.\d+)?)\s*(lb|kg)\b", re.IGNORECASE)
UNIT_PRICE_RE = re.compile(r"(?<!\d)(\d+\.\d{2})\s*/\s*(lb|kg)\b", re.IGNORECASE)

def extract_first_price(s: str) -> Optional[float]:
    m = PRICE_RE.search(s.replace(" ", ""))
    if not m:
        m = PRICE_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def looks_like_price_line(s: str) -> bool:
    # True if contains something like 3.99 possibly with trailing letters
    return extract_first_price(s) is not None and len(s.strip()) <= 16

# ---- Pairing: name line + price line ----
def pair_items_from_ocr_lines(ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build minimal structured candidates from OCR lines:
    - pairs "name" with next "price" line
    - also preserves special weight lines (qty / unit_price) if present
    Output:
      [{"name_raw": "...", "line_total": 3.99}, ...]
    """
    pairs: List[Dict[str, Any]] = []
    last_name: Optional[str] = None

    # We'll also keep a small rolling window for weight items:
    # If we see name + total + (weight x) + (unit price /lb), we attach those hints.
    recent_name: Optional[str] = None
    recent_total: Optional[float] = None
    recent_weight_qty: Optional[float] = None
    recent_unit_price: Optional[float] = None

    def flush_weight_candidate():
        nonlocal recent_name, recent_total, recent_weight_qty, recent_unit_price
        if recent_name and recent_total is not None:
            obj = {"name_raw": recent_name, "line_total": recent_total}
            if recent_weight_qty is not None:
                obj["qty_hint"] = recent_weight_qty
            if recent_unit_price is not None:
                obj["unit_price_hint"] = recent_unit_price
            pairs.append(obj)
        recent_name = None
        recent_total = None
        recent_weight_qty = None
        recent_unit_price = None

    for ln in ocr_lines:
        t = normalize_money((ln.get("text") or "").strip())
        if is_noise_line(t):
            continue

        # Weight hints
        m_w = WEIGHT_RE.search(t.replace("1b", "lb").replace("Ib", "lb"))
        m_u = UNIT_PRICE_RE.search(t.replace("1b", "lb").replace("Ib", "lb"))

        # If this is a unit price line like "2.29/lb"
        if m_u:
            try:
                recent_unit_price = float(m_u.group(1))
            except Exception:
                pass
            continue

        # If this is a weight qty line like "2.42 lb x"
        if m_w and ("x" in t.lower() or "×" in t):
            try:
                recent_weight_qty = float(m_w.group(1))
            except Exception:
                pass
            continue

        # If looks like a price line, pair with last name
        if looks_like_price_line(t):
            price = extract_first_price(t)
            if last_name and price is not None:
                pairs.append({"name_raw": last_name, "line_total": price})
                last_name = None
            # Also treat as possible total line for weight candidate (name often appears before)
            if recent_name and recent_total is None and price is not None:
                recent_total = price
            continue

        # Otherwise treat as name line
        # For weight candidates, we store name to attach later if we see the pattern
        last_name = t
        # Start/refresh weight candidate name
        # (we flush previous if it was complete)
        if recent_name and recent_total is not None:
            flush_weight_candidate()
        recent_name = t

    # flush any remaining weight candidate
    if recent_name and recent_total is not None:
        flush_weight_candidate()

    return pairs

# ---- Build LLM input text ----
def build_llm_input_text(ocr_lines: List[Dict[str, Any]]) -> str:
    """
    Keep only lines that are likely useful for receipt parsing,
    but still include date/total lines if present.
    """
    keep: List[str] = []
    for ln in ocr_lines:
        t = normalize_money((ln.get("text") or "").strip())
        if not t:
            continue

        tl = t.lower()

        # Keep potential date/time lines
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", t) or re.search(r"\b\d{1,2}:\d{2}\b", t):
            keep.append(t)
            continue

        # Keep total-like lines (e.g., 66.66)
        if looks_like_price_line(t) and len(t) <= 8:
            keep.append(t)
            continue

        # Keep item-like lines: either name-ish or price-ish
        if not is_noise_line(t):
            keep.append(t)

    # De-duplicate consecutive duplicates (OCR often repeats)
    compact: List[str] = []
    prev = None
    for t in keep:
        if t == prev:
            continue
        compact.append(t)
        prev = t
    return "\n".join(compact)

# ---- Postprocess structured output ----
def postprocess(structured: Dict[str, Any]) -> Dict[str, Any]:
    def _strip(v):
        return v.strip() if isinstance(v, str) else v

    structured["merchant"] = _strip(structured.get("merchant"))
    structured["datetime"] = _strip(structured.get("datetime"))

    items = structured.get("items") or []
    for it in items:
        it["name_raw"] = _strip(it.get("name_raw"))
        it["name"] = _strip(it.get("name"))
        if isinstance(it.get("category"), str):
            it["category"] = it["category"].strip()

        # clamp confidence
        c = it.get("confidence")
        if isinstance(c, (int, float)):
            it["confidence"] = max(0.0, min(1.0, float(c)))

    totals = structured.get("totals") or {}
    structured["totals"] = totals

    return structured


def dedup_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cands:
        key = (c.get("name_raw"), c.get("line_total"), c.get("qty_hint"), c.get("unit_price_hint"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
