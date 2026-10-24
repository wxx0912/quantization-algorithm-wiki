---
{
  "id": "repo-awq",
  "title": "mit-han-lab/llm-awq — 官方实现（auto_scale 与 auto_clip）",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61",
  "version": "d6e797a42b9ef7778de8ee2352116e0f48a78d61",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "awq/quantize/auto_scale.py: auto_scale_block（内含嵌套函数 _search_module_scale）、apply_scale；awq/quantize/auto_clip.py: auto_clip_layer、auto_clip_block",
  "topic": "reconstruction",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

# mit-han-lab/llm-awq — 官方实现（auto_scale 与 auto_clip）

## 这份来源是什么

AWQ 官方实现，用于核对 scale 搜索的候选集合与裁剪的比较口径，是 `algorithm-awq` 的源码依据。

## 已读定位与阅读形式

固定 commit 下读取 GitHub 原始文件，未克隆运行。

## 关键事实

- `_search_module_scale` 使用 20 个候选（`ratio=0,0.05,…,0.95`），并对 scale 做数值归一化。
- `auto_clip_layer` 在每个输出通道、每个 weight group 上比较局部输出误差以选择裁剪阈值。
- `auto_clip_block` 跳过名称匹配 Q/K 的投影，因为 QK 矩阵乘法使单独 clipping 的误差代理不够直接。
- README 区分搜索、fake quant 与 real quant 的不同入口。

## 不能推出的结论

- 代码候选集合是论文区间的实现细节，不能反过来当作论文已经测试端点 1。
- 静态源码不证明目标运行时接受其 packing 或启用低比特 kernel。
- 未运行 fake/real quant，也未核对导出产物与后端格式的一致性。
