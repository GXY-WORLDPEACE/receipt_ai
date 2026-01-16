# Receipt AI (OCR + DeepSeek)

## 1) Install
conda activate receipt_ai
pip install -r requirements.txt

## 2) Set env

export DEEPSEEK_API_KEY="sk-ecf36b22e19d4c75a321e7739aae7761"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
export OCR_LANG="german"

uvicorn app.main:app --host 0.0.0.0 --port 8000


## 3) Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## 4) Test
curl -X POST "http://localhost:8000/v1/receipt/parse?currency=EUR" \
  -H "accept: application/json" \
  -F "file=@/path/to/receipt.jpg"
