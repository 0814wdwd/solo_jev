# SOLO Decision

**Structured Input Layout Optimizer for Decision Models**

把完整记录交给决策模型之前，先把行和字段排好，让服务端复用更多前缀计算。
输入支持 Pandas、NumPy、JSON / JSONL；输出按原始行顺序返回决策和概率。

![完整记录重排与前缀复用](../assets/layout-overview.svg)

## 快速使用

在本项目源码目录中安装客户端：

```bash
python -m pip install ".[pandas,hub]"
```

准备一个 JEV-9B 服务后：

```python
from pathlib import Path
from solo_decision import DecisionEngine

with DecisionEngine() as engine:
    result = engine.scan(Path("tickets.jsonl"), "Does the customer request a monetary refund?")
print(result.to_pandas())
```

同一个 `scan` 方法可以直接接收 DataFrame。每条请求仍包含完整记录，模型端通过
前缀缓存复用计算。排序不会删除字段，结果会还原到输入的行位置。
嵌套 JSON 保留对象与数组结构；JSONL 会先读入内存，再规划全局布局。
[输入契约](inputs.md) · [API](api.md)

不需要 GPU 的离线体验：

```bash
python examples/offline_layout.py
```

已有兼容 Linux NVIDIA GPU 主机时，可以运行：

```bash
bash deploy/jev9b.sh up
```

参考服务配置已在单张 RTX 4090 24 GB 上实测：JEV-9B BF16、16K 上下文、并发 4。
完整依赖和部署边界见[部署说明](deployment.md)。客户端无需安装模型权重或 Torch。

## 两个演示分别说明什么

**重复上下文。** 输入里 ID 在最前，后面有很多重复的政策或账户字段。把这些字段
移到前面，并把相似行排列在一起，可以增加服务端复用。演示主动调整重复字段长度，
比较 80%、97%、98.3% 的可复用字段值 token 占比；这不是缓存命中率。
三档实测相对原序分别提升 **1.49×、7.90×、11.71×**。最长上下文一档中，128 条记录
从 **274.9 秒降到 23.5 秒**；每条完整输入的最大长度为 15,611 token。

![重复上下文实测](../assets/shared-benchmark.svg)

**相关性。** 所有字段的 NDV 都是 8，长度都是 128 token。单看每列的不同值个数无法
区分顺序，而 SOLO 会利用列之间的联合前缀结构。128 行的两轮实测中，SOLO 相对
原序提高 **3.17×**，相对 NDV 排序提高 **3.28×**，从 4.15 提高到 13.64 行/秒。

![相关字段实测](../assets/correlated-benchmark.svg)

两个演示都是合成负载，在真实 RTX 4090 / vLLM 服务上运行。时间包含规划、请求处理
和首次缓存填充；这些简单的退款判断任务实测准确率为 100%。它们展示特定结构下的
收益，不代表任意生产任务都能达到同样的倍数或准确率。

[完整实验协议、原始数据和复现命令](benchmarks.md) · [英文项目首页](../README.md)

## 看自己的数据是否适合

```python
from solo_decision import LayoutOptimizer

report = LayoutOptimizer().explain(your_dataframe)
print(report.to_pandas())
```

`explain` 在本地报告字段 NDV、前缀组合数和相邻记录的共享前缀字节。
实际吞吐还取决于 tokenizer、缓存块大小、显存、并发和模型，因此要用 `engine.compare`
对自己的数据测量，并同时检查准确率和原序结果一致率。

核心整数规划实现保留了上一轮 SOLO PR 的优化：编码后的贪心分组阶段为 O(NM²)，
固定列序的行分组为 O(NM)。[实现与来源](architecture.md)
