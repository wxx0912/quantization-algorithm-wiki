# 基础样章来源

这里的“已读”指核对所列段落或代码，不表示已运行算法。原论文与当前仓库快照可能不一致。

| ID | 类型 | URL | 版本与已读定位 |
| --- | --- | --- | --- |
| `paper-gptq` | paper | https://arxiv.org/html/2210.17323v2 | §3–4，Algorithm 1，§5，附录 A.2；全文 HTML，重点核对重构、Cholesky、校准与性能范围 |
| `repo-gptq` | code | https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf | gptq.py：add_batch、fasterquant；quant.py：Quantizer.find_params、quantize、Quant3Linear；固定 HEAD 快照 |
| `paper-smoothquant` | paper | https://arxiv.org/html/2211.10438v7 | §4 Eq.3–4，§5.1–5.3，Table 2–4；全文 HTML |
| `repo-smoothquant` | code | https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5 | smoothquant/smooth.py：smooth_ln_fcs、smooth_ln_fcs_llama_like、smooth_lm |
| `paper-quarot` | paper | https://arxiv.org/html/2404.00456v2 | §3.4、§4 Stage 1a–2c、§5、附录 A.1/A.7/A.10；全文 HTML |
| `repo-quarot` | code | https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a | README、fake_quant/main.py、fake_quant/rotation_utils.py；这里只核对参考变换与量化编排，未运行 CUDA |
| `paper-omniquant` | paper | https://arxiv.org/html/2308.13137v3 | §3 Eq.1–5、§4.1、Table 1、附录训练预算；全文 HTML |
| `repo-omniquant` | code | https://github.com/OpenGVLab/OmniQuant/tree/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14 | README、quantize/omniquant.py、quantize/quantizer.py；LWC/LET 优化、MSE、原位融合 |
