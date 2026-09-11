# Image Insight

**🌐 [English](README.md) | [中文](README.zh.md)**

Give the computer a photo and see how much it can understand — a productized demo of a case-property visual identification PoC. **By default, runs entirely offline on local hardware — no network calls, no cloud API of any kind.** An opt-in remote-GPU backend is also available for hardware with no local GPU (see "Local vs. remote GPU backend" below) — using it means photos leave the machine, so it must be pointed at infrastructure you control, not a public API.

## Two modes

- **Fast mode**: open-vocabulary recognition — item names are freely described by the model (not limited to a fixed category list) and auto-classified into the system's official taxonomy of 12 major categories / 84 subcategories (see `特别需求补充.md` §3). Answers only "what's there and how many"; does not cover brand/model/text details. Multiple photos of the same item from different angles can be uploaded and are automatically merged into one result (taking the max count across angles, not the sum).
- **Advanced mode**: structured understanding — major/sub category, brand, model, color, visible text, appearance features, condition. Every field is tagged as one of "confirmed / suspected / undetermined"; nothing is ever fabricated when uncertain. Appearance notes describe only "what is visually observed" (color, shape, location) and never draw diagnostic conclusions about the cause of marks or stains — that's the job of a professional forensic examiner. Visible text is first read by Qwen3-VL and then cross-checked locally by PaddleOCR ("the VLM understands, the OCR reads precisely"). Also supports auto-merging multiple angle photos of the same item.

Both modes report a confidence score; when it's too low the item is flagged "needs verification" rather than forcing a definite category (the threshold is configurable via environment variables, see `.env.example`). The 12 mandatory-manual-review subcategories listed in `特别需求补充.md` §2 (collectibles, surveillance recording equipment, drugs, etc.) are always flagged in red in the UI regardless of confidence.

Both modes share the same locally-loaded Qwen3-VL model — only the prompt differs.

The UI supports batch upload: selected photos show thumbnails immediately (each can be removed individually); files whose names follow the "item-id_timestamp-hash" pattern are auto-merged as multiple angles of the same item, the rest are analyzed independently — one upload can handle several unrelated items at once, no need to submit separately.

## Data persistence

Every analysis result is automatically saved to a local SQLite database (`image_insight.db`, path configurable via the `IMAGE_INSIGHT_DB_PATH` environment variable), keyed by "item id (goods_id) + mode (fast/advanced)". **Re-analyzing the same item overwrites the existing record rather than adding a new one** — the database stores "this item's latest analysis result," not an append-only log of every run. Fast-mode and advanced-mode results are stored separately, so running fast mode won't overwrite a prior advanced-mode record, and vice versa.

A save failure does not affect the display of the current analysis result (same as OCR — a nice-to-have, not a hard dependency). `GET /api/records` (optionally filtered with `?mode=fast` or `?mode=advanced`) shows what's currently stored.

## Local vs. remote GPU backend

Both modes call Qwen3-VL through a small `ChatCompletionTransport` abstraction (`image_insight/vlm/qwen_client.py`), so which backend actually runs the model is a config choice, not a code change:

- **`QWEN_BACKEND=local`** (default) — loads Qwen3-VL's weights on this machine's own GPU/CPU. Fully offline at request time, as described above.
- **`QWEN_BACKEND=remote`** — for running the app itself on hardware with no GPU (e.g. a company desktop): each photo is sent to `QWEN_REMOTE_BASE_URL`, an OpenAI-compatible chat-completions endpoint (e.g. [vLLM](https://github.com/vllm-project/vllm), SGLang, or Xinference serving Qwen3-VL) running on a GPU box elsewhere. Optional `QWEN_REMOTE_API_KEY` is sent as a Bearer token if the endpoint requires one.

**This is the one place the "no photo leaves the machine" guarantee doesn't hold** — switching to `remote` means photo bytes travel over the network to whatever `QWEN_REMOTE_BASE_URL` points at. Only point it at infrastructure your organization controls (a private server on your own VPN/intranet), never a public third-party API, if the photos are case-property evidence subject to `特别需求补充.md` §1.6's no-external-upload requirement. See the `QWEN_BACKEND` / `QWEN_REMOTE_*` comments in `.env.example` for every variable.

On a GPU-less host that only ever uses `QWEN_BACKEND=remote`, you can skip installing `torch`/`torchvision`/`rfdetr`/`transformers`/`accelerate`/`qwen-vl-utils` from `requirements.txt` (only needed for the local backend) and swap `paddlepaddle-gpu` for the CPU-only `paddlepaddle` package if you still want the OCR cross-check.

## Quick start

```bash
pip install -r requirements.txt
# See the comments in requirements.txt: torch and paddlepaddle-gpu need to be installed
# from their respective CUDA-specific package indexes first

python -m uvicorn image_insight.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The first run automatically downloads the Qwen3-VL-2B-Instruct weights (~4GB), cached by default under this project's own `models/huggingface/` folder rather than the user-wide `~/.cache/huggingface` — so the whole app (code + model) stays one self-contained, movable folder. Only needed once — fully offline afterward. To use a different cache location, or reuse a checkpoint a teammate already downloaded, see the `HF_HOME` / `QWEN_MODEL_ID` notes in `.env.example`.

Stop the service with `Ctrl+C` in the terminal.

### Running tests

```bash
python -m pytest tests/ -q
```

### Environment variables (optional)

Copy `.env.example` to `.env` and adjust as needed — it's loaded automatically at startup (python-dotenv), no manual `export`/`setx` required. Sensible defaults are used if unset, and by default (`QWEN_BACKEND=local`) advanced mode requires no API key at all (everything runs locally) — see "Local vs. remote GPU backend" above for the opt-in exception. A variable already set in your real shell/OS environment always takes priority over the value in `.env`.

## Directory layout

```
image_insight/          application code
  vlm/                   local Qwen3-VL inference wrapper (shared by both modes)
  api/                   FastAPI service + frontend page
  taxonomy.py            official 12-category / 84-subcategory taxonomy + classification validation, mandatory-review list
  ocr/                    PaddleOCR wrapper (advanced-mode visible-text cross-check)
  db.py                   local SQLite persistence (overwrite-on-save, keyed by goods_id+mode)
  config.py               runtime configuration (reads environment variables)
  detection/               RF-DETR zero-shot phone counting (legacy artifact, not wired into the live service, see below)
tests/                    unit/integration tests (136 cases, pytest)
eval/                     real-world accuracy evaluation of RF-DETR phone counting (14/15, 93.3%)
物品图片/                  132 real item photos (case-property evaluation dataset)
Image-Insight-开发计划.md  the original phased development plan document

The following are project background reference materials for the overall
case-property "one item, one file" scheme (not part of this demo's code):
涉案财物一物一档视觉可信管理产品需求与必要性分析_V0.2.docx / .md
涉案财物全生命周期视觉可信管理技术方案_V0.2.docx / .md
特别需求补充.md
task for andy.docx
模型选择.docx
Mage-VL业务场景与产品化研究任务书.docx   (a separate, unrelated research track)
```

## Known notes

- `rf-detr-medium.pth` (~400MB) is the locally cached RF-DETR pretrained weight file — it is not junk, please don't delete it. If deleted, it will be re-downloaded the next time `image_insight/detection/` is used (currently not wired into the live service, kept only as an already-evaluated legacy artifact).
- Fast mode originally used RF-DETR/COCO's 5 fixed categories, later switched to open-vocabulary local-VLM recognition based on feedback; `image_insight/detection/phone_counter.py` is kept as an independently-verified module (93.3% accuracy), but the current service no longer calls it.
- An 8GB-VRAM GPU can run Qwen3-VL-2B smoothly; with more headroom, you can swap `QWEN_MODEL_ID` in `.env.example` for `Qwen/Qwen3-VL-4B-Instruct` for better results — no code changes needed.
- The confidence score is Qwen3-VL's own subjective rating from the same JSON output, not a calibrated probability like a classifier's (generative models have no native softmax confidence) — it's used only for ranking and the "needs verification" threshold, and the frontend never presents it to users as a percentage.
- Advanced mode's multi-angle merging buckets by "major category" (`ItemAnalysis` has no free-text item name to match against, unlike fast mode): if the same photo happens to contain two different items of the exact same major category, multi-angle merging may mistakenly combine them into one record. This matches the intended use case of "one goods_id = one physical item, just photographed from different angles," but isn't suited to a genuinely independent multi-item, multi-angle scenario.
- A single model generation has a length cap (`QWEN_MAX_NEW_TOKENS`, default 1536); when a photo has many items or long descriptions, the output JSON may get cut off before it's finished (the UI will report a parse error like "Unterminated string"). `image_insight/vlm/json_repair.py` salvages the items that were already fully generated from the truncated raw text rather than discarding everything just because the last entry is incomplete; whenever truncation happens, the result is explicitly labeled "content truncated, items may be missing" and forced into manual review — it's never silently shown as a normal result. If this warning shows up often, increase `QWEN_MAX_NEW_TOKENS` in `.env.example` (at the cost of slower analysis).
- This PoC demo is not a production business system; all results are for reference only, with human confirmation as the final word.
