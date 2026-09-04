# Mini MetaAttention

基于 PyTorch 的 CPU Attention demo，包含：

- Phase 1 Q-only tiled attention
- Phase 1.5 Q+K tiled online softmax
- Phase 2.1 Parallel Attention DSL 与 Python code generator
- `tile_q × tile_k` CPU 自动调优与性能图表
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
python -m pip install -r requirements.txt

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

运行自动调优实验：

```bash
python -m experiments.autotune
```

结果生成：

- `results/search.csv`：全部 tile candidate 结果
- `results/compare.csv`：reference、fixed 64×64 与 tuned 对比
- `results/best_causal_softmax.json`：各 workload 的最佳配置
- `plots/search.png`：tile 搜索热力图
- `plots/throughput.png`：吞吐量对比图
- `generated/tuned_causal_softmax.py`：代表 workload 的调优生成代码
