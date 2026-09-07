# Image Insight

**🌐 [简体中文](#简体中文) | [English](#english)**

<a id="简体中文"></a>

给计算机一张照片，看它能看懂多少——涉案财物视觉识别 PoC 的产品化 demo。**完全本地离线运行，不联网、不调用任何云端 API。**

## 两种模式

- **快速模式**：开放词汇识别，物品名称由模型自由描述（不局限于固定类别列表），并自动归类到系统的 12 大类 / 84 细类官方分类体系（见 `特别需求补充.md` §3）。只回答"有什么、各有几个"，不涉及品牌/型号/文字细节。可上传同一物品的多张不同角度照片，系统会自动合并为一个结果（取各角度计数的最大值，而不是求和）。
- **高级模式**：结构化理解——大类/细类、品牌、型号、颜色、可见文字、外观特征、成色。每个字段都标注"确认/疑似/无法确定"三种状态之一；不确定的地方绝不编造。外观描述只写"看到了什么"（颜色、形状、位置），不对痕迹/污渍的成因下诊断性结论，那是专业鉴定人员的工作）。可见文字由 Qwen3-VL 初判后再经本地 PaddleOCR 交叉校验（"VLM 负责看懂，OCR 负责看清字"）。同样支持上传同一物品的多张角度照片自动合并。

两种模式都会给出置信度，过低时标记为"待核实"，不强行给出确定类别（阈值可通过环境变量配置，见 `.env.example`）；`特别需求补充.md` §2 列出的 12 个强制人工核实细类（收藏品、监控录像设备、毒品等）无论置信度高低都会在界面强制标红提示。

两种模式共用同一个本地加载的 Qwen3-VL 模型，只是提示词不同。

界面支持批量上传：选中的照片会立即显示缩略图（可单独移除），文件名符合"物品编号_时间戳哈希"格式的自动合并为同一物品的多角度结果，其余各自独立分析——一次上传可以同时处理多件不相关的物品，不需要分开提交。

## 数据持久化

每次分析结果会自动存入本地 SQLite 数据库（`image_insight.db`，`IMAGE_INSIGHT_DB_PATH` 环境变量可改路径），按"物品编号（goods_id）+ 模式（快速/高级）"为主键。**同一件物品第二次分析会覆盖原有记录，不会新增一条**——数据库存的是"这件物品目前最新的分析结果"，不是每次运行都往里堆的流水记录。快速模式和高级模式的结果分开存，跑一次快速模式不会覆盖之前的高级模式记录，反之亦然。

保存失败不会影响本次分析结果的正常显示（和 OCR 一样是"锦上添花"，不是硬依赖）。`GET /api/records`（可加 `?mode=fast` 或 `?mode=advanced` 过滤）能看到当前存了什么。

## 快速开始

```bash
pip install -r requirements.txt
# 见 requirements.txt 内注释：torch 和 paddlepaddle-gpu 需要先从各自的 CUDA 专用源安装

python -m uvicorn image_insight.api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。首次启动会自动下载 Qwen3-VL-2B-Instruct 权重（约 4GB，缓存到 `~/.cache/huggingface`，只需一次，此后完全离线运行）。

停止服务：终端里 `Ctrl+C`。

### 运行测试

```bash
python -m pytest tests/ -q
```

### 环境变量（可选）

复制 `.env.example` 并按需修改；不设置的话都有合理默认值，高级模式不需要任何 API key（全部本地跑）。

## 目录说明

```
image_insight/          应用代码
  vlm/                   Qwen3-VL 本地推理封装（快速+高级模式共用）
  api/                   FastAPI 服务 + 前端页面
  taxonomy.py            12 大类 / 84 细类官方分类表 + 归类校验、强制核实名单
  ocr/                    PaddleOCR 文字识别封装（高级模式的可见文字交叉校验）
  db.py                   本地 SQLite 持久化（按 goods_id+模式 覆盖式保存分析结果）
  config.py               运行配置（读环境变量）
  detection/               RF-DETR 零样本手机计数（历史产物，现在没接入正式服务，见下）
tests/                    单元/集成测试（136 个用例，pytest）
eval/                     RF-DETR 手机计数的真实准确率评估（14/15，93.3%）
物品图片/                  132 张真实物品照片（涉案财物评估数据集）
Image-Insight-开发计划.md  最初的分阶段开发计划文档

以下为项目背景资料（涉案财物"一物一档"整体方案的参考文档，非本 demo 代码的一部分）：
涉案财物一物一档视觉可信管理产品需求与必要性分析_V0.2.docx / .md
涉案财物全生命周期视觉可信管理技术方案_V0.2.docx / .md
特别需求补充.md
task for andy.docx
模型选择.docx
Mage-VL业务场景与产品化研究任务书.docx   （另一条独立研究线，与本 demo 无关）
```

## 已知情况

- `rf-detr-medium.pth`（约 400MB）是 RF-DETR 预训练权重的本地缓存文件，不是垃圾文件，请勿删除——删除后下次用到 `image_insight/detection/`（目前未接入正式服务，仅作为已评估的历史产物保留）会重新下载。
- 快速模式最初基于 RF-DETR/COCO 的 5 个固定类别，后按反馈改为开放词汇的本地 VLM 识别；`image_insight/detection/phone_counter.py` 保留作为一个已验证过准确率（93.3%）的独立模块，但当前服务不再调用它。
- 8GB 显存的显卡可以流畅跑 Qwen3-VL-2B；如果显存更宽裕、想要更好的效果，可以把 `.env.example` 里的 `QWEN_MODEL_ID` 换成 `Qwen/Qwen3-VL-4B-Instruct`，不需要改代码。
- 置信度分数是 Qwen3-VL 自己在同一次 JSON 输出里给出的主观打分，不是分类器那种经过校准的概率（生成式模型没有原生 softmax 置信度）——仅用于排序和"待核实"阈值判断，前端也不会把它当百分比展示给用户。
- 高级模式的多角度合并按"大类"分桶（`ItemAnalysis` 没有像快速模式那样的自由文本物品名可用于匹配）：如果同一张照片里恰好有两个完全同大类的不同物品，多角度合并可能会误并为一条记录。这符合"一个 goods_id 对应一件实物、只是分角度拍摄"的实际使用场景，但不适合真正独立的多物品多角度场景。
- 模型单次生成有长度上限（`QWEN_MAX_NEW_TOKENS`，默认 1536），照片里物品较多或描述较长时，输出的 JSON 可能还没写完就被截断（界面会报"Unterminated string"一类的解析错误）。`image_insight/vlm/json_repair.py` 会从截断的原始文本里抢救出已经完整生成的物品，不会因为最后一条不完整就把前面全部丢弃；只要发生截断，结果会明确标注"内容被截断，可能遗漏物品"并强制人工核实，不会当作正常结果悄悄展示。如果频繁遇到这个提示，可以把 `.env.example` 里的 `QWEN_MAX_NEW_TOKENS` 调大（代价是单次分析变慢）。
- 本 PoC 演示程序不是正式业务系统，所有结果仅供参考，最终以人工确认为准。

---

<a id="english"></a>

## English

Give the computer a photo and see how much it can understand — a productized demo of a case-property visual identification PoC. **Runs entirely offline on local hardware — no network calls, no cloud API of any kind.**

### Two modes

- **Fast mode**: open-vocabulary recognition — item names are freely described by the model (not limited to a fixed category list) and auto-classified into the system's official taxonomy of 12 major categories / 84 subcategories (see `特别需求补充.md` §3). Answers only "what's there and how many"; does not cover brand/model/text details. Multiple photos of the same item from different angles can be uploaded and are automatically merged into one result (taking the max count across angles, not the sum).
- **Advanced mode**: structured understanding — major/sub category, brand, model, color, visible text, appearance features, condition. Every field is tagged as one of "confirmed / suspected / undetermined"; nothing is ever fabricated when uncertain. Appearance notes describe only "what is visually observed" (color, shape, location) and never draw diagnostic conclusions about the cause of marks or stains — that's the job of a professional forensic examiner. Visible text is first read by Qwen3-VL and then cross-checked locally by PaddleOCR ("the VLM understands, the OCR reads precisely"). Also supports auto-merging multiple angle photos of the same item.

Both modes report a confidence score; when it's too low the item is flagged "needs verification" rather than forcing a definite category (the threshold is configurable via environment variables, see `.env.example`). The 12 mandatory-manual-review subcategories listed in `特别需求补充.md` §2 (collectibles, surveillance recording equipment, drugs, etc.) are always flagged in red in the UI regardless of confidence.

Both modes share the same locally-loaded Qwen3-VL model — only the prompt differs.

The UI supports batch upload: selected photos show thumbnails immediately (each can be removed individually); files whose names follow the "item-id_timestamp-hash" pattern are auto-merged as multiple angles of the same item, the rest are analyzed independently — one upload can handle several unrelated items at once, no need to submit separately.

### Data persistence

Every analysis result is automatically saved to a local SQLite database (`image_insight.db`, path configurable via the `IMAGE_INSIGHT_DB_PATH` environment variable), keyed by "item id (goods_id) + mode (fast/advanced)". **Re-analyzing the same item overwrites the existing record rather than adding a new one** — the database stores "this item's latest analysis result," not an append-only log of every run. Fast-mode and advanced-mode results are stored separately, so running fast mode won't overwrite a prior advanced-mode record, and vice versa.

A save failure does not affect the display of the current analysis result (same as OCR — a nice-to-have, not a hard dependency). `GET /api/records` (optionally filtered with `?mode=fast` or `?mode=advanced`) shows what's currently stored.

### Quick start

```bash
pip install -r requirements.txt
# See the comments in requirements.txt: torch and paddlepaddle-gpu need to be installed
# from their respective CUDA-specific package indexes first

python -m uvicorn image_insight.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The first run automatically downloads the Qwen3-VL-2B-Instruct weights (~4GB, cached to `~/.cache/huggingface`, only needed once — fully offline afterward).

Stop the service with `Ctrl+C` in the terminal.

#### Running tests

```bash
python -m pytest tests/ -q
```

#### Environment variables (optional)

Copy `.env.example` and adjust as needed; sensible defaults are used if unset, and advanced mode requires no API key at all (everything runs locally).

### Directory layout

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

### Known notes

- `rf-detr-medium.pth` (~400MB) is the locally cached RF-DETR pretrained weight file — it is not junk, please don't delete it. If deleted, it will be re-downloaded the next time `image_insight/detection/` is used (currently not wired into the live service, kept only as an already-evaluated legacy artifact).
- Fast mode originally used RF-DETR/COCO's 5 fixed categories, later switched to open-vocabulary local-VLM recognition based on feedback; `image_insight/detection/phone_counter.py` is kept as an independently-verified module (93.3% accuracy), but the current service no longer calls it.
- An 8GB-VRAM GPU can run Qwen3-VL-2B smoothly; with more headroom, you can swap `QWEN_MODEL_ID` in `.env.example` for `Qwen/Qwen3-VL-4B-Instruct` for better results — no code changes needed.
- The confidence score is Qwen3-VL's own subjective rating from the same JSON output, not a calibrated probability like a classifier's (generative models have no native softmax confidence) — it's used only for ranking and the "needs verification" threshold, and the frontend never presents it to users as a percentage.
- Advanced mode's multi-angle merging buckets by "major category" (`ItemAnalysis` has no free-text item name to match against, unlike fast mode): if the same photo happens to contain two different items of the exact same major category, multi-angle merging may mistakenly combine them into one record. This matches the intended use case of "one goods_id = one physical item, just photographed from different angles," but isn't suited to a genuinely independent multi-item, multi-angle scenario.
- A single model generation has a length cap (`QWEN_MAX_NEW_TOKENS`, default 1536); when a photo has many items or long descriptions, the output JSON may get cut off before it's finished (the UI will report a parse error like "Unterminated string"). `image_insight/vlm/json_repair.py` salvages the items that were already fully generated from the truncated raw text rather than discarding everything just because the last entry is incomplete; whenever truncation happens, the result is explicitly labeled "content truncated, items may be missing" and forced into manual review — it's never silently shown as a normal result. If this warning shows up often, increase `QWEN_MAX_NEW_TOKENS` in `.env.example` (at the cost of slower analysis).
- This PoC demo is not a production business system; all results are for reference only, with human confirmation as the final word.
