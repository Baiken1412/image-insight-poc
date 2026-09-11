# Image Insight

**🌐 [English](README.md) | [中文](README.zh.md)**

给计算机一张照片，看它能看懂多少——涉案财物视觉识别 PoC 的产品化 demo。**默认完全本地离线运行，不联网、不调用任何云端 API。** 也提供可选的远程 GPU 后端，供没有本地 GPU 的部署环境使用（见下文"本地 vs 远程 GPU 后端"）——启用后照片会离开本机，因此必须指向自己可控的基础设施，而不是公网第三方 API。

## 两种模式

- **快速模式**：开放词汇识别，物品名称由模型自由描述（不局限于固定类别列表），并自动归类到系统的 12 大类 / 84 细类官方分类体系（见 `特别需求补充.md` §3）。只回答"有什么、各有几个"，不涉及品牌/型号/文字细节。可上传同一物品的多张不同角度照片，系统会自动合并为一个结果（取各角度计数的最大值，而不是求和）。
- **高级模式**：结构化理解——大类/细类、品牌、型号、颜色、可见文字、外观特征、成色。每个字段都标注"确认/疑似/无法确定"三种状态之一；不确定的地方绝不编造。外观描述只写"看到了什么"（颜色、形状、位置），不对痕迹/污渍的成因下诊断性结论，那是专业鉴定人员的工作）。可见文字由 Qwen3-VL 初判后再经本地 PaddleOCR 交叉校验（"VLM 负责看懂，OCR 负责看清字"）。同样支持上传同一物品的多张角度照片自动合并。

两种模式都会给出置信度，过低时标记为"待核实"，不强行给出确定类别（阈值可通过环境变量配置，见 `.env.example`）；`特别需求补充.md` §2 列出的 12 个强制人工核实细类（收藏品、监控录像设备、毒品等）无论置信度高低都会在界面强制标红提示。

两种模式共用同一个本地加载的 Qwen3-VL 模型，只是提示词不同。

界面支持批量上传：选中的照片会立即显示缩略图（可单独移除），文件名符合"物品编号_时间戳哈希"格式的自动合并为同一物品的多角度结果，其余各自独立分析——一次上传可以同时处理多件不相关的物品，不需要分开提交。

## 数据持久化

每次分析结果会自动存入本地 SQLite 数据库（`image_insight.db`，`IMAGE_INSIGHT_DB_PATH` 环境变量可改路径），按"物品编号（goods_id）+ 模式（快速/高级）"为主键。**同一件物品第二次分析会覆盖原有记录，不会新增一条**——数据库存的是"这件物品目前最新的分析结果"，不是每次运行都往里堆的流水记录。快速模式和高级模式的结果分开存，跑一次快速模式不会覆盖之前的高级模式记录，反之亦然。

保存失败不会影响本次分析结果的正常显示（和 OCR 一样是"锦上添花"，不是硬依赖）。`GET /api/records`（可加 `?mode=fast` 或 `?mode=advanced` 过滤）能看到当前存了什么。

## 本地 vs 远程 GPU 后端

两种模式都通过一个很薄的 `ChatCompletionTransport` 抽象（`image_insight/vlm/qwen_client.py`）调用 Qwen3-VL，因此实际用哪种后端跑模型只是配置项，不需要改代码：

- **`QWEN_BACKEND=local`**（默认）——在本机自己的 GPU/CPU 上加载 Qwen3-VL 权重，分析时完全离线，即上文所述。
- **`QWEN_BACKEND=remote`**——用于把本应用部署在没有 GPU 的机器上（比如公司台式机）：每张照片会发送给 `QWEN_REMOTE_BASE_URL` 指定的、OpenAI 兼容的 chat-completions 接口（例如用 [vLLM](https://github.com/vllm-project/vllm)、SGLang 或 Xinference 在别处的 GPU 服务器上跑 Qwen3-VL）。如果远程接口需要鉴权，可选的 `QWEN_REMOTE_API_KEY` 会作为 Bearer token 一起发送。

**这是唯一一处"照片不出本机"这条保证不成立的地方**——切换到 `remote` 意味着照片字节会经网络发送到 `QWEN_REMOTE_BASE_URL` 指向的地方。如果这些照片属于涉案财物证据、受 `特别需求补充.md` §1.6 不外传要求约束，就只能把它指向自己单位可控的基础设施（自有 VPN/内网上的私有服务器），绝不能指向公网第三方 API。每个变量的具体说明见 `.env.example` 里 `QWEN_BACKEND` / `QWEN_REMOTE_*` 相关注释。

如果一台机器只会用 `QWEN_BACKEND=remote`、不打算跑本地后端，可以不装 `requirements.txt` 里的 `torch`/`torchvision`/`rfdetr`/`transformers`/`accelerate`/`qwen-vl-utils`（只有本地后端需要），如果还想保留 OCR 交叉校验，把 `paddlepaddle-gpu` 换成纯 CPU 版的 `paddlepaddle` 即可。

## 快速开始

```bash
pip install -r requirements.txt
# 见 requirements.txt 内注释：torch 和 paddlepaddle-gpu 需要先从各自的 CUDA 专用源安装

python -m uvicorn image_insight.api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。首次启动会自动下载 Qwen3-VL-2B-Instruct 权重（约 4GB，默认缓存到项目根目录下的 `models/huggingface/`，而不是用户目录下的 `~/.cache/huggingface`——这样整个应用文件夹拷贝/搬走时模型也一起带走；只需下载一次，此后完全离线运行。想换成别的缓存位置，或者同事已经下载好模型直接复用，见 `.env.example` 里的 `HF_HOME` / `QWEN_MODEL_ID` 说明）。

停止服务：终端里 `Ctrl+C`。

### 运行测试

```bash
python -m pytest tests/ -q
```

### 环境变量（可选）

复制 `.env.example` 为 `.env` 并按需修改，启动时会自动加载（python-dotenv），不用手动 `export`/`setx`；不设置的话都有合理默认值，默认（`QWEN_BACKEND=local`）情况下高级模式不需要任何 API key（全部本地跑）——可选的例外见上文"本地 vs 远程 GPU 后端"。系统/终端里已经设置过的同名环境变量优先级高于 `.env` 文件里的值。

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
