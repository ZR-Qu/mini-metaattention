# Mini MetaAttention

Mini MetaAttention 的 CPU Attention demo，包含 Q-only tiled 和 Q+K tiled online softmax 实现

当前支持：

- causal self-attention
- `float32`
- CPU
- 输入 shape：`[B, H, S, D]`
- Q+K tiled online softmax

## 使用

```bash
source .venv/bin/activate

python bench.py --threads 1
python bench.py --threads 8
python -m pytest -q
```

benchmark 默认使用：
```
B=1, H=4, S=1024, D=64
tile_q=16,32,64,128,256
online tile_q=64, tile_k=16,32,64,128,256,512,1024
```
最后输出 Q-only 和 online 配置的延迟、误差和 score 大小，并给出各自最快的 tile
