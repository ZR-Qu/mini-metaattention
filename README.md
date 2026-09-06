# Mini MetaAttention

基于 PyTorch 的 CPU Attention code-generation 与 tile autotuning demo

Mini MetaAttention 采用统一的 AttentionSpec 表达不同 Attention 计算模式，通过 generator 自动生成 tiled Attention 实现，并利用 autotuner 根据 workload 搜索最佳 `tile_q × tile_k` 配置

支持：

* causal softmax
* ReLU Attention
* causal ReLU
* unseen primitive composition

输入格式：

```text
Q: [B, H, Sq, D]
K: [B, H, Sk, D]
V: [B, H, Sk, Dv]
```

使用 CPU `float32`；示例包含 causal softmax、ReLU 和 causal ReLU



## Dev Container

VS Code中安装 Microsoft Dev Containers 扩展，打开仓库后执行：

```text
Dev Containers: Reopen in Container
```

1. 一键运行测试：

```bash
./scripts/test.sh
```

2. 使用 generator 生成 Attention 实现：

```bash
python -m miniattn.generate examples.causal_softmax \
  --tile-q 64 \
  --tile-k 64 \
  --output generated/causal_softmax.py

python -m miniattn.generate examples.relu \
  --tile-q 64 \
  --tile-k 64 \
  --output generated/relu.py
```


3. Causal Softmax 自动调优，生成 CSV、JSON、热力图、吞吐量图：

```bash
python -m experiments.autotune_causal_softmax
```

4. 不同 Attention 语义下的 tile autotuning 对比：

```bash
python -m experiments.autotune_variants \
  --label large \
  --batch 1 --heads 8 --seq-len 1024 \
  --dim 128 --value-dim 128 --threads 8 \
  --tile-candidates 16 32 64 128 256 512 1024 \
  --warmups 5 --repeats 20
```

5. 单个 model-derived workload：

```bash
python -m experiments.run_autotune \
  --label deepseek-s2048 \
  --variant causal_softmax \
  --batch 1 \
  --heads 16 \
  --seq-len 2048 \
  --dim 192 \
  --value-dim 128 \
  --threads 8 \
  --tile-candidates 16 32 64 128 256 512 1024 2048 \
  --warmups 5 \
  --repeats 20
```

仓库中已保存 DeepSeek-derived、LLaMA-style 和 ViT-style 实验结果



## 本地安装与测试

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
./scripts/test.sh
```



## Code Generation

生成 Attention 实现：

```bash
python -m miniattn.generate examples.causal_softmax \
  --tile-q 64 --tile-k 64 \
  --output generated/causal_softmax.py

python -m miniattn.generate examples.relu \
  --tile-q 64 --tile-k 64 \
  --output generated/relu.py

python -m miniattn.generate examples.causal_relu \
  --tile-q 64 --tile-k 64 \
  --output generated/causal_relu.py
```

Generator 根据 AttentionSpec 中定义的 primitives 进行 lowering，生成对应实现

生成模块提供：

```python
attention(q, k, v)
```



## Autotuning 实验

1. 比较不同 Attention 语义下的最佳 tile：

```bash
python -m experiments.autotune_variants \
  --label large \
  --batch 1 --heads 8 --seq-len 1024 \
  --dim 128 --value-dim 128 --threads 8 \
  --tile-candidates 16 32 64 128 256 512 1024 \
  --warmups 5 --repeats 20
```

输出：

```text
results/<run-id>/
    variants_search.csv   # 候选csv
    variants_compare.csv  # 对比csv
    run.json              # 本轮参数

plots/<run-id>/
    variants.png          # 性能对比图
```

2. 运行单个 workload configuration：

```bash
python -m experiments.run_autotune \
  --label deepseek-s2048 \
  --variant causal_softmax \
  --heads 16 --seq-len 2048 \
  --dim 192 --value-dim 128 \
  --threads 8 \
  --tile-candidates 16 32 64 128 256 512 1024 2048 \
  --warmups 5 --repeats 20
```

输出：

```text
results/<run-id>/
    search.csv            # 全部tile候选结果
    compare.csv           # 对比csv
    run.json              # 本轮参数

plots/<run-id>/
    comparison.png        # 性能对比图
```

3. Causal Softmax 自动调优

```bash
python -m experiments.autotune_causal_softmax
```

输出：

```
results/causal-softmax/
plots/causal-softmax/
generated/tuned_causal_softmax.py
```


- `results/causal-softmax/search.csv`：全部 tile 候选结果
- `results/causal-softmax/compare.csv`：对比csv
- `results/causal-softmax/best_causal_softmax.json`：各 workload 最佳配置
- `plots/causal-softmax/search.png`：tile 搜索热力图
- `plots/causal-softmax/throughput.png`：吞吐量对比图
- `generated/tuned_causal_softmax.py`：调优代码



## Architecture

```
AttentionSpec
      |
      v
 Generator
      |
      v
Generated tiled Attention
      |
      v
 Autotuner
      |
      v
Best tile configuration
```
generator 不依赖预定义 variant 名称
