# 本地模式强力透视回正实施计划

> 依据：`docs/superpowers/specs/2026-08-18-local-strong-perspective-correction-design.md`
>
> 范围：只增强本地离线处理。不修改 GPT Image 2 路径、批处理上限、输出格式或下载交互。

## 实施原则

- 按测试驱动的小步骤实施：每个任务先加失败测试，再写最小实现，然后运行定向测试和相关回归。
- 几何分析只读取清理后的主体蒙版；不从 RGB 内部的门缝、密码盘或装饰线估计外框。
- 强力回正成功时，RGB 和蒙版各执行一次 `warpPerspective`；不先旋转再透视。
- 强力回正失败时才进入现有保守旋转。已经回正时返回原数组，避免无意义重采样。
- 新阈值集中为模块级常量，并用合成测试固定验收边界。
- 保留现有 `rotation_*` metrics 用于回退审计，新增 `perspective_*` metrics 作为几何流程的主状态。

## 任务 1：建立外轮廓与四边稳健拟合

**文件：**

- 新建 `tests/test_local_perspective_correction.py`
- 新建 `rembg/perspective.py`

### 1.1 先写外框检测失败测试

在新测试文件中加入可复用的蒙版生成器，覆盖：

- 轴对齐圆角矩形能产生顺序为左上、右上、右下、左下的四角。
- 轻微旋转和轻度梯形蒙版能拟合出四条外边。
- 圆角、少量外轮廓毛刺与顶部小缺口不影响主要边线。
- 有效内点跨度不足 45%、内点比例不足 55%或拟合残差超过 1.2% 时返回低置信原因码。
- 缺边、非凸轮廓、短边、相邻边夹角小于 60° 或交点超出外扩 8% 范围时不返回可用四边形。

运行并确认因尚无实现而失败：

```bash
python3 -m unittest tests.test_local_perspective_correction.OuterFrameDetectionTests -v
```

### 1.2 实现结构化检测结果

在 `rembg/perspective.py` 中增加冻结 dataclass：

- `EdgeFit`：存储规一化直线参数、角度、内点比例、有效跨度和归一化残差。
- `OuterFrameEstimate`：存储四条边、`float32` 四角、主体包围盒、整体置信度和原因码。

原因码使用稳定的英文字符串，至少包含 `ok`、`no_contour`、`insufficient_edge_support`、`invalid_intersection`、`invalid_quad` 和 `incomplete_frame`，便于测试与 metrics 审计。

### 1.3 实现拟合辅助函数

增加：

- `largest_outer_contour(mask)`：用保守阈值二值化，只返回最大外轮廓和包围盒。
- `_edge_band_points(contour, bbox, side)`：用包围盒相对宽度抽取上、下、左、右边带点，并排除圆角端区主导拟合。
- `_robust_fit_edge(points, side, subject_size)`：使用 `cv2.fitLine` 和迭代残差剔除，最终计算支持度、跨度和残差。
- `_line_intersection(first, second)`：对近平行线做数值安全检查。
- `estimate_outer_frame(mask)`：组合四边，计算四角，检查顺序、凸性、面积、边长、夹角和 8% 交点范围。

实现中所有跨度、残差和外扩阈值都使用主体包围盒的宽高归一化，不使用特定分辨率的像素常量（数值稳定的极小值除外）。

### 1.4 验证并提交

```bash
python3 -m unittest tests.test_local_perspective_correction.OuterFrameDetectionTests -v
git add rembg/perspective.py tests/test_local_perspective_correction.py
git commit -m "Add robust outer-frame detection"
```

## 任务 2：计算安全的目标矩形和单次变换计划

**文件：**

- 修改 `tests/test_local_perspective_correction.py`
- 修改 `rembg/perspective.py`

### 2.1 先写变换规划失败测试

新增 `PerspectivePlanTests`，覆盖：

- 目标宽度来自上下边长的稳健平均，高度来自左右边长的稳健平均，不套用固定商品比例。
- 目标矩形中心与检测四边形中心一致。
- 边缘主体在缩小比例 `0.95 <= scale < 1.0` 时可完整保留，计划被允许。
- 需要 `scale < 0.95` 才能避免越界时，返回 `clipping_risk`。
- 最大角点位移超过主体长边 8% 或预计面积比不在 0.85–1.15 内时，返回 `excessive_warp`。
- 最大归一化位移小于 0.5% 且所有边离目标轴小于 0.5° 时返回 `not_needed`。
- 已回正的输入返回原 RGB/蒙版对象，不调用 OpenCV warp。

### 2.2 实现变换计划

增加 `PerspectivePlan` 冻结 dataclass，存储源四角、目标四角、单应性矩阵、归一化位移、面积比、统一缩放比例、是否需要变换和原因码。

增加：

- `_edge_axis_deviation(edge, side)`：将水平边与 0°、垂直边与 90° 比较。
- `target_rectangle(frame)`：以四边形中心和两组对边长度生成轴对齐矩形。
- `_project_contour(contour, matrix)`：用实际外轮廓而不是只用包围盒评估越界。
- `build_perspective_plan(frame, contour, canvas_shape)`：生成单应性矩阵，如有必要则围绕主体中心叠加最多 5% 的统一缩放，并执行位移、面积和边界门槛。

缩放需直接合并进同一个 3×3 矩阵，不另外调用 resize 或 affine warp。

### 2.3 实现单次重采样

在 `rembg/perspective.py` 中实现 `apply_perspective_plan(image, mask, plan)`，并在接入时删除 `rembg/product_image.py` 中未使用的 `_safe_perspective_correction`：

- RGB：`cv2.INTER_LANCZOS4`，边界填充 `(255, 255, 255)`。
- 蒙版：`cv2.INTER_LINEAR`，边界填充 `0`。
- 输出 `dsize` 始终为原图宽高。

用 mock 断言强力回正成功时只有两次 `warpPerspective`（RGB 和蒙版各一次），且不调用 `warpAffine`。

### 2.4 验证并提交

```bash
python3 -m unittest tests.test_local_perspective_correction.PerspectivePlanTests -v
git add rembg/perspective.py tests/test_local_perspective_correction.py
git commit -m "Plan safe single-pass perspective transforms"
```

## 任务 3：将强力回正和保守旋转编排为一个几何流程

**文件：**

- 修改 `tests/test_local_perspective_correction.py`
- 修改 `tests/test_original_size_fidelity.py`
- 修改 `rembg/product_image.py`
- 修改 `rembg/perspective.py`（仅在决策测试暴露计划结果缺失时）

### 3.1 先写分支和审计结果失败测试

新增 `GeometryDecisionTests`，通过合成图像和定向 patch 覆盖六个稳定状态：

- 可信且需要回正：`perspective_status=applied`，步骤列表包含强力透视回正，不运行普通旋转。
- 可信但无需回正：`perspective_status=not_needed`，无 warp，无额外 warning。
- 外框不可信且普通旋转成功：`perspective_status=fallback_rotation`，记录回退角度与支持度。
- 外框不可信且无可信旋转：`perspective_status=skipped_low_confidence`；仅在存在可见风险时加入“外框检测不完整，建议人工检查”并置 `review`。
- 强力回正需要超过 5% 缩小或变形过大：尝试普通旋转；仍有风险时 `perspective_status=skipped_clipping_risk` 并置 `review`。
- `correct_geometry=False`：`perspective_status=disabled` 且 `rotation_status=disabled`。

修改现有旋转流程测试：通过 patch 强制外框检测低置信，明确它们测的是“普通旋转回退”，而不再假定旋转是首选路径。

### 3.2 实现统一决策函数

在 `rembg/product_image.py` 中增加 `_correct_geometry(image, mask, *, border_contact_ratio)`，复用 `_mask_quality` 已计算的边界接触指标，并返回变换后的 RGB、蒙版、步骤、警告和 metrics。内部顺序固定为：

1. 提取最大外轮廓并拟合四边。
2. 如外框可信，生成强力回正计划。
3. 计划为 `applied` 时执行单次透视变换；计划为 `not_needed` 时直接返回原数组。
4. 外框不可信或计划被安全门槛拒绝时，调用现有 `_rotation_angle` 和 `_rotate_pair`。
5. 根据原因码、旋转置信和可见几何偏差决定是否加 warning，避免把“已经回正”标记为失败。具体来说，任一已拟合外边偏轴达 0.5°、已检测角点位移达 0.5%、`border_contact_ratio > 0.008` 或安全计划明确返回 `clipping_risk`/`excessive_warp` 时，视为存在可见复核风险；单纯的 `no_contour` 且无边界接触时仍由现有蒙版质量警告决定状态。

可见风险判定必须依据已有的边线角度/偏移、蒙版边界接触或回退旋转结果，不得将所有低置信都升级为 `review`。

### 3.3 接入 `process_product_image`

将现有内联旋转分支替换为 `_correct_geometry` 调用，并合并其步骤、警告和 metrics。保证：

- 强力回正仍发生在高光控制和白底合成之前。
- `working_image` 和 `working_mask` 保留回正后对齐的数据，不破坏人工蒙版修正。
- `ProcessingResult.status` 仍由最终 warnings 决定；正常 `applied` 和 `not_needed` 不自动生成 warning。

### 3.4 完整记录 metrics

使用标量 key 记录四角，避免扩大当前 `dict[str, float | str]` 的序列化类型：

- `perspective_status`、`perspective_reason`、`perspective_confidence`。
- `perspective_top_degrees`、`perspective_bottom_degrees`、`perspective_left_degrees`、`perspective_right_degrees`。
- 每条边的 `*_inlier_ratio`、`*_span_ratio` 和 `*_residual_ratio`。
- `perspective_corner_tl_x_ratio` / `*_y_ratio` 到 `perspective_corner_bl_*`，以画布宽高归一化。
- `perspective_max_displacement_ratio`、`perspective_area_ratio`、`perspective_scale`。
- `rotation_status`、`rotation_degrees`、`rotation_line_support`，用于普通旋转回退。

### 3.5 验证并提交

```bash
python3 -m unittest tests.test_local_perspective_correction.GeometryDecisionTests -v
python3 -m unittest tests.test_original_size_fidelity -v
git add rembg/product_image.py rembg/perspective.py tests/test_local_perspective_correction.py tests/test_original_size_fidelity.py
git commit -m "Integrate safe perspective correction into local processing"
```

## 任务 4：增加端到端几何和画质验收

**文件：**

- 修改 `tests/test_local_perspective_correction.py`
- 修改 `tests/test_original_size_fidelity.py`

### 4.1 增加合成图端到端测试

通过“轴对齐原型图 + 已知透视矩阵”生成 RGB/蒙版对，不依赖外部图片。覆盖：

- 轻微旋转矩形回正后，上下边残余角度不超过 0.35°，左右边同理。
- 轻度梯形回正后，任一组对边的角度差不超过 0.35°。
- RGB 中添加密码盘、门缝和装饰线，而蒙版外轮廓不变；回正角点不应受内部直线影响。
- 轴对齐输入的 RGB 和蒙版逐像素保持不变，证明无重采样。
- 每个输出宽高与输入严格一致；主体中心偏移在归一化容差内；`perspective_scale >= 0.95`。
- 光栅化后的软蒙版仍存在部分透明边缘，且最终白底合成无黑边或越界主体。

### 4.2 增加 4032×3024 尺寸回归

用简单矩形蒙版和低细节 RGB 合成 4032×3024 输入，验证输出仍为 4032×3024。该测试只验证画布不变式，避免使用巨大复杂纹理造成不必要的测试开销。

### 4.3 验证并提交

```bash
python3 -m unittest tests.test_local_perspective_correction -v
python3 -m unittest tests.test_original_size_fidelity -v
git add tests/test_local_perspective_correction.py tests/test_original_size_fidelity.py
git commit -m "Cover perspective correction geometry and fidelity"
```

## 任务 5：验证批处理审计、直接下载与 GPT 路径隔离

**文件：**

- 修改 `tests/test_direct_downloads.py`
- 修改 `tests/test_openai_image.py`（仅在需要增强隔离断言时）
- 如测试暴露审计丢失，修改 `rembg/product_batch.py`

### 5.1 增加批处理回归

测试一个 `processed`、一个 `review` 和一个 `failed` 项目的计数和下载路径，并检查：

- manifest 完整序列化 `perspective_status`、原因、四边支持度、角点、位移、面积变化和缩放比例。
- 强力回正成功的本地项目仍生成可直接下载的 JPG/JPEG/PNG。
- 回退警告使文件进入 `review/`，但不阻止其它项目输出。
- 几何变换后的 `working_image` 和 `working_mask` 尺寸对齐，原有人工蒙版删除/恢复流程仍可使用。

`product_batch.py` 当前会原样序列化 `ProcessingResult.metrics`，因此预期不需要生产代码改动；只在失败测试证明存在丢失时做最小修复。

### 5.2 锁定 GPT Image 2 隔离

在 `tests/test_openai_image.py` 增加或保留断言：当 `processing_engine="openai"` 时不调用 `_correct_geometry`，生成提示词、请求次数、尺寸恢复和人工复核状态不变。

### 5.3 验证并提交

```bash
python3 -m unittest tests.test_direct_downloads -v
python3 -m unittest tests.test_openai_image -v
git add tests/test_direct_downloads.py tests/test_openai_image.py rembg/product_batch.py
git commit -m "Verify perspective audit and batch regressions"
```

如 `rembg/product_batch.py` 或 `tests/test_openai_image.py` 最终没有内容变化，不将它们加入提交。

## 任务 6：全量回归和用户样例验收

**文件：**

- 不预设生产代码改动。
- 若样例是用户私有图片，只本地验证，不加入 Git。

### 6.1 运行全量自动测试

在已安装 NumPy、OpenCV 和 Pillow 的项目 Python 环境中运行：

```bash
python3 -m unittest discover -s tests -v
```

当前 macOS 系统 `python3` 缺少 `numpy`、`cv2` 和 `PIL`，不能用它作为回归环境；实施时应使用项目原有运行环境或先建立隔离的虚拟环境。不为本功能额外更改依赖版本，除非现有环境无法运行已有测试。

### 6.2 用橙色保险柜原图验收

使用与生产一致的本地模型运行用户样例，记录并核对：

- 输出严格为 4032×3024。
- `perspective_status` 与实际分支一致；如应用回正，两组对边基本平行且接近目标轴。
- 主体中心仍在原画面区域，`perspective_scale >= 0.95`，蒙版无越界。
- 密码盘数字、锁具、划痕和漆面纹理均来自原图像素，无生成或修补。
- 如原图客观缺边，结果应安全回退并给出具体复核提示，不强制拉伸。

保存一份输出和 manifest 用于用户复核，但除非用户明确要求，不提交图片产物。

### 6.3 最终检查

```bash
git status --short
git log --oneline -6
```

确认工作树只包含预期变更，每个中间提交都可独立理解和回滚。若样例验收暴露阈值问题，先添加可复现的合成测试，再调整数值稳定性细节；不放宽设计文档中 5% 缩放、8% 位移和 0.85–1.15 面积边界。

## 完成判定

只有以下条件全部满足时才完成实施：

- 本地 `correct_geometry=True` 默认优先运行外轮廓四边回正。
- 强力回正中 RGB 只重采样一次，输出画布尺寸不变。
- 已回正输入不重采样，不可信输入安全回退。
- 5% 缩放下限、8% 最大位移、0.85–1.15 面积比与低置信回退均有自动测试。
- 审计 metrics、warnings、`processed/review/failed` 计数与实际分支一致。
- JPG、JPEG、PNG 直接下载、人工蒙版修正和 GPT Image 2 回归测试通过。
- 橙色保险柜样例通过视觉与指标验收，或在客观缺边时返回设计中规定的安全复核结果。
