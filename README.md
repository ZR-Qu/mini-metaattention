# Mini MetaAttention

Mini MetaAttention阶段1的demo，用来比较不同 `tile_q` 下的运行时间和中间 score 大小

当前支持：

- causal self-attention
- `float32`
- CPU
- 输入 shape：`[B, H, S, D]`

## 使用

```bash
source .venv/bin/activate

python bench.py --threads 1
python bench.py --threads 8
```

benchmark 默认使用：
```
B=1, H=4, S=1024, D=64
tile_q=16,32,64,128,256
```
最后输出显示每个 tile_q 的延迟、误差和 score 大小，并给出最快的 tile
