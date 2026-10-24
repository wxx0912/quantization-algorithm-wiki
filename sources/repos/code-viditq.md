---
{
  "id": "code-viditq",
  "title": "A-suozhang/ViDiT-Q — 官方实现（量化层与基础量化器）",
  "category": "repo",
  "url": "https://github.com/A-suozhang/ViDiT-Q/tree/a8add8be17ad91069aba7a7209aa88e04b2fe41e",
  "version": "a8add8be17ad91069aba7a7209aa88e04b2fe41e",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README、quant_utils/qdiff/viditq/viditq_quant_layer.py、quant_utils/qdiff/base/base_quantizer.py、kernels/README.md",
  "topic": "mixed-architecture",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# A-suozhang/ViDiT-Q — 官方实现（量化层与基础量化器）

## 这份来源是什么

ViDiT-Q 官方实现，用于核对量化层的实现层次与基础量化器结构，是 `guide-architecture-transfer`、`guide-deployment-evidence` 与 `case-paper-evidence` 的源码依据。

## 已读定位与阅读形式

克隆固定 commit 后读取 README、量化层与基础量化器文件、`kernels/` 目录；仅静态核验，未编译或运行。

## 关键事实

- `viditq_quant_layer.py` 的 `ViDiTQuantizedLinear` 采用静态权重量化 + 动态逐 token 激活量化，并含通道掩码缩放与 Hadamard 旋转路径；`base_quantizer.py` 提供 `BaseQuantizer`/`StaticQuantizer`/`DynamicQuantizer`。
- `kernels/` 提供 CUDA 扩展（`kernels/README.md` 只含安装、benchmark 与致谢三节，未写覆盖范围或限制）；量化层的旋转路径引用 `qdiff/quarot` 的 Hadamard 实现。
- 主 README 注明 Flux 模型「not fully supported all vidtq techniques」，且 paper 侧的版本与该 commit 的对应关系未在本仓库注明。

## 不能推出的结论

- 代码存在不等于本项目已编译或运行 kernel；未做任何 GPU 验证。
- 该 commit 与论文 v2 的版本对应关系未在仓库注明，不能据 commit 推断「已包含论文之后的新能力」，也不能与论文配置混写。
