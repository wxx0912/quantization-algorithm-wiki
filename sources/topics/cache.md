# Cache 与长上下文

KV cache 量化的来源。这一组回答“把 attention 的键值缓存单独压到低比特”时的分布特征、量化维度（per-channel / per-token）、离群与窗口策略及其证据边界。与权重/激活量化分开：cache 是流式张量，量化器要兼顾在线实现与后端 kernel。

本主题的来源清单与引用计数见 [../INDEX.generated.md](../INDEX.generated.md)。
