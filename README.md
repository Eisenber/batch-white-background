# 批量生成白底图

把商品照片（保险柜等）批量转成**纯白底电商图**，同时保持**原尺寸、原位置、原比例**——不裁切、不缩放、不移动主体。基于 [rembg](https://github.com/danielgatis/rembg) 扩展，在原抠图能力之上叠加了一整套面向电商商品图的本地校正流水线，并提供 Web 界面与命令行两种入口。

## 功能特性

- **原尺寸原位置合成**：输出与源图分辨率完全一致，主体像素、中心位置、比例保持不变。
- **两种处理引擎**（详见下文「处理引擎」）：
  - `local` —— 本地分割模型 + 多步校正，无需联网、不产生按张费用。
  - `openai` —— 调用 GPT Image 2 整图重生成，效果更好但需配置 API 密钥。
- **自动方向与颜色标准化**：EXIF 方向纠正、ICC 转 sRGB。
- **几何校正**：主体原位回正（直线检测自动纠偏，靠近边缘时自动跳过以避免裁切）。
- **瑕疵处理**：顶部杂物清理、保纹理高光压制、蒙版边缘收紧。
- **人工复核与修正**：可疑结果进入「待复核」，支持在界面上涂抹删除/恢复蒙版后重新出图。
- **审计信息**：每张图输出执行步骤、告警、指标（角度、支持度、高光占比等），可追溯。

## 环境要求

- Python 3.12（本仓库使用 `.venv` 虚拟环境）
- 依赖库：OpenCV、NumPy、Pillow、onnxruntime、Gradio/FastAPI 等（见上游 rembg 依赖）
- 可选（OpenAI 引擎）：`openai`、`python-dotenv`（见 `requirements-openai.txt`）

## 安装

```bash
# 创建并激活虚拟环境（示例，按需调整）
python3.12 -m venv .venv
source .venv/bin/activate

# 安装核心依赖（rembg 上游依赖）
pip install "rembg[cpu,cli]"

# 若使用 OpenAI 引擎，再安装：
pip install -r requirements-openai.txt
```

> 本仓库是 rembg 的源码扩展版本，直接运行源码即可；`rembg` 目录位于项目根目录下，可被直接导入。

## 快速开始

启动 Web 界面（批量白底图工作台）：

```bash
# 直接运行源码（本仓库未安装为 console script，故通过 -c 调用 click 入口）
.venv/bin/python -c "from rembg.cli import main; main()" s
```

启动后：

- 界面地址：<http://localhost:7000>
- API 文档：<http://localhost:7000/api>

在界面里选择图片、处理引擎（本地 / OpenAI）、画质与输出格式，即可批量处理。处理结果分目录保存：

- `processed/` —— 正常出图
- `review/` —— 需人工复核（存在告警，如过曝、疑似顶部杂物、裁切风险）
- `.edit/` —— 用于人工蒙版修正的中间资产（源图 + 蒙版）

> 本地分割模型（如 `birefnet-massive`）首次使用会自动下载到 `~/.u2net/`，也可用 `rembg d` 预下载。

## 命令行

`rembg` 命令提供以下子命令：

| 命令 | 说明 |
|------|------|
| `b`  | 字节流输入（按固定宽高逐帧读取） |
| `d`  | 下载模型 |
| `i`  | 单文件抠图 |
| `p`  | 文件夹批量抠图 |
| `s`  | 启动 HTTP 服务 + Web 界面（本项目的白底图工作台） |

示例：

```bash
# 启动服务（默认端口 7000、监听 0.0.0.0）
.venv/bin/python -c "from rembg.cli import main; main()" s -p 7000

# 只启动 API，不带界面（降低空闲 CPU 占用）
.venv/bin/python -c "from rembg.cli import main; main()" s --no-ui

# 预下载模型
.venv/bin/python -c "from rembg.cli import main; main()" d birefnet-massive
```

## 处理引擎

### 本地引擎（`local`）

不联网、不按张计费。先由分割模型产出主体蒙版，再执行本地校正流水线：

1. 方向与颜色标准化
2. 主体分割（高质量用 `birefnet-massive`，快速用 `u2net`）
3. 保险柜顶部杂物清理
4. 主体原位回正
5. 保纹理高光压制
6. 蒙版边缘收紧
7. 原尺寸原位置白底合成

`BatchOptions` 中 `quality` 决定模型：`high` → `birefnet-massive`，`fast` → `u2net`。

### OpenAI 引擎（`openai`）

调用 GPT Image 2 生成整张白底商品图，并恢复到源图尺寸。生成结果一律标记为「待复核」，因文字、按键、锁具、表面细节需人工确认。该模式每批最多 10 张，并会按张产生费用。

## OpenAI 模式配置

在项目根目录创建 `.env`（参考）：

```dotenv
# 图片生成 API（GPT Image 2）
IMG_BASE_URL=https://your-relay-endpoint.example.com/v1
IMG_MODEL=gpt-image-2
IMG_API_KEY=sk-xxxxxxxxxxxxxxxx
```

| 变量 | 说明 |
|------|------|
| `IMG_BASE_URL` | OpenAI 兼容接口的 base URL |
| `IMG_MODEL` | 使用的生成模型名 |
| `IMG_API_KEY` | API 密钥 |

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

当前全部通过：`26 passed`。

## 目录结构

```
.
├── rembg/                  # 核心源码（rembg 扩展）
│   ├── cli.py              # click 入口，注册各子命令
│   ├── product_image.py    # 单图处理流水线（校正、合成）
│   ├── product_batch.py    # 批量编排、清单、人工修正
│   ├── product_ui.py       # Gradio 白底图工作台
│   ├── openai_image.py     # GPT Image 2 生成
│   ├── commands/           # b/d/i/p/s 子命令
│   └── sessions/           # 分割模型会话（birefnet-massive 等）
├── tests/                  # 单元测试
├── requirements-openai.txt # OpenAI 引擎额外依赖
└── .env                    # OpenAI 模式配置（可选，勿提交）
```

## 注意事项

- **OpenCV 5 兼容性**：`cv2.HoughLinesP` 在 OpenCV 5.0 中将返回形状从 `(N, 1, 4)` 改为 `(N, 4)`。本项目已做兼容处理（`lines.reshape(-1, 4)`），并附回归测试；若你在别处复用该逻辑，请留意此差异。
- **API 安全**：`.env` 含密钥，已被 `.gitignore` 排除，切勿提交。
- **过曝告警**：当检测到大面积过曝且纹理无法可靠恢复时，结果会进入「待复核」而非自动降级处理。
