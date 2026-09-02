# Mini MetaAttention

基于 PyTorch 的 CPU Attention demo，包含：

- Phase 1 Q-only tiled attention
- Phase 1.5 Q+K tiled online softmax
- Phase 2.1 Parallel Attention DSL 与 Python code generator
- causal softmax 和 ReLU Attention 示例

生成代码支持 CPU、float32 输入：

```text
Q: [B, H, Sq, D]
K: [B, H, Sk, D]
V: [B, H, Sk, Dv]
```

## 使用

```bash
source .venv/bin/activate

python -m pytest -q
python bench.py --threads 1
python bench.py --threads 8
```

生成 Phase 2.1 代码：

```bash
python -m miniattn.generate examples.causal_softmax \
  --tile-q 64 --tile-k 64 \
  --output generated/causal_softmax.py

python -m miniattn.generate examples.relu \
  --tile-q 64 --tile-k 64 \
  --output generated/relu.py
```

生成模块统一提供：

```python
attention(q, k, v)
```
