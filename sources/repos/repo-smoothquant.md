---
{
  "id": "repo-smoothquant",
  "title": "mit-han-lab/smoothquant — 官方实现（α 平滑与 W8A8 量化入口）",
  "category": "repo",
  "url": "https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5",
  "version": "c61476d728e42ae0d8a35e7e78494edcac3237b5",
  "version_kind": "commit",
  "retrieved_at": "2026-10-24",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "smoothquant/smooth.py: smooth_ln_fcs、smooth_ln_fcs_llama_like、smooth_lm；smoothquant/calibration.py: get_act_scales、get_static_decoder_layer_scales；smoothquant/fake_quant.py: W8A8Linear、quantize_model/quantize_opt/quantize_llama_like/quantize_mixtral/quantize_falcon",
  "topic": "core",
  "excerpt_sha256": null
}
---
## 这份来源是什么

SmoothQuant 官方实现，用于核对 α 平滑公式、多模块成组方式、校准统计口径与 W8A8 各粒度入口，是 `algorithm-smoothquant` 的源码依据。

## 已读定位与阅读形式

固定 commit 下静态读取 `smoothquant/` 的 `smooth.py`、`calibration.py`、`fake_quant.py`（未读 `smoothquant/opt.py` 的 INT8 模块与 `examples/` 脚本）。源码为本地快照（无 `.git`），commit 号取自 URL。未运行校准、量化或评测。

## 关键事实

- **平滑公式**（`smooth_ln_fcs`）：`weight_scales` 取同组各 `fc` 逐输入通道 `|w|` 的最大值再跨 `fc` 取 max、`clamp(min=1e-5)`；`scales = (act_scales^α / weight_scales^(1−α)).clamp(min=1e-5)`；然后 `ln.weight.div_(scales)`、`ln.bias.div_(scales)`、每个 `fc.weight.mul_(scales.view(1,-1))`。`smooth_ln_fcs_llama_like` 同式，但只对 RMSNorm 的 `weight` 相除（RMSNorm 无 bias），断言类型为 `LlamaRMSNorm`/`MistralRMSNorm`/`MixtralRMSNorm`。
- **成组方式**（`smooth_lm` 按 layer 类型分派）：OPT/Bloom 为 `attn_ln`→qkv、`ffn_ln`→fc1；Llama/Mistral 为 `input_layernorm`→{q,k,v}、`post_attention_layernorm`→{gate_proj, up_proj}；Falcon 在 `not module.config.new_decoder_architecture and module.config.parallel_attn` 时把 `attn_ln` 的下游合并成 `[qkv, fc1]` 一组，其余配置走 else，并按 `new_decoder_architecture` 二选一：为真时 `ln_attn→qkv`、`ln_mlp→fc1`，为假时 `input_layernorm→qkv`、`post_attention_layernorm→fc1`；Mixtral 把 `ffn_ln` 的下游合并成 `block_sparse_moe.gate` 加全部专家的 `w1`/`w3`。
- **校准**（`calibration.py`）：`get_act_scales(num_samples=512, seq_len=512)` 用 forward hook 对模型内**所有** `nn.Linear` 的输入通道取 |激活| 的逐样本 max、再取跨样本 max（不限于最终被平滑的层）；`get_static_decoder_layer_scales` 另统计每层输入/输出的整体 abs max，其键名 `model.decoder.layers.{i}...` 硬编码 OPT 结构，当前仅被 `examples/export_int8_model.py` 调用，未见于 Bloom 路径。
- **W8A8 各粒度入口**（`fake_quant.py`）：INT8 absmax 权重提供 `per_channel`/`per_tensor`，激活提供 `per_token`/`per_tensor`；`W8A8Linear.from_float(module, weight_quant, act_quant, quantize_output=False)` 组合出不同粒度（论文里的 O1/O2/O3 是外部术语，仓库内无同名实现，此对应关系为本页推断）。模块级 `quantize_model` 默认 `weight_quant='per_channel'`、`act_quant='per_token'`；`quantize_opt` 默认改为 per_tensor/per_tensor 且 `quantize_bmm_input=True`；`quantize_falcon` 用 per_channel/per_token 且 `quantize_bmm_input=True`。
- **职责分离**：`smooth.py` 只改写归一化与线性层参数，不含 INT8 GEMM 或 kernel；INT8 执行在 `smoothquant/opt.py`（`Int8OPTForCausalLM` 等，依赖 `torch_int` 的 W8A8 线性、量化 LayerNorm 与量化 bmm）；`examples/` 下是导出、校准与评测脚本（`export_int8_model.py` 调用 `Int8OPTForCausalLM`）。两处实现细节本页未读。

## 不能推出的结论

- 该文件只做平滑与参数改写，不含 INT8 GEMM/kernel；融合前后的浮点等价性、以及 `quantize_bmm_input` 等开关的实际影响本页未验证。
- 成组规则（尤其把 Mixtral 的 router 与全部专家合成一组、把 Falcon 的 qkv 与 fc1 合成一组）是参考实现的取舍，不能证明任意 MoE、共享专家或分叉结构都可套用，也不代表论文实验都用了这些分组。
- α 的函数默认值固定为 0.5，与论文 OPT/BLOOM 的取值相同；论文对 GLM-130B 用 0.75、对 Llama-2/Falcon/Mistral 用 0.6–0.9（Table 7），不能把函数默认值当作所有模型的配置。
- 校准默认 `512×512` 与按 max 统计不是论文唯一设置；本页未运行校准或量化，也未核对统计对结果的影响。
- 同一仓库里 OPT 默认 per_tensor 激活、Llama 类默认 per_token，说明粒度由具体入口决定，不能只凭算法名推断粒度。
