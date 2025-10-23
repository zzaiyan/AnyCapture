# AnyCapture
[![AnyCapture](https://badge.fury.io/py/anycapture.svg)](https://badge.fury.io/py/anycapture)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Downloads](https://img.shields.io/pypi/dm/anycapture.svg)](https://pypi.org/project/anycapture/)
[![Python Version](https://img.shields.io/pypi/pyversions/anycapture.svg)](https://pypi.org/project/anycapture/)

AnyCapture是一个Python工具库，专门用于捕获函数执行过程中的局部变量。该库主要致力于解决深度学习模型中间结果提取的技术难题，特别适用于深度学习模型中Attention Map的可视化分析。

## ✨ 核心特性
- 🚀 **多变量捕获**：支持通过装饰器同时捕获多个局部变量
- 📦 **字典缓存**：变量以结构化字典形式存储，便于管理和访问
- 🧹 **缓存管理**：提供clear()方法进行缓存清理
- 🔄 **队列功能**：支持限制缓存大小，自动管理内存使用

## 背景与动机
在深度学习模型可视化过程中，开发者经常遇到以下技术挑战：

**传统解决方案的局限性：**
* **返回值传递法**：需要修改模型结构，将嵌套在模型深处的Attention Map逐层返回，在训练时又需要还原代码
* **全局变量法**：使用全局变量直接记录Attention Map，容易在训练时遗忘修改导致内存溢出

这些问题在实际开发中普遍存在，严重影响了开发效率。

**PyTorch Hook机制的技术限制：**

虽然PyTorch提供了hook机制来获取中间结果：
```python
handle = net.conv2.register_forward_hook(hook)
```

但在实际应用中存在以下技术障碍：

以Vision Transformer为例，其典型结构如下：
```python
class VisionTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        ...
        self.blocks = nn.Sequential(*[Block(...) for i in range(depth)])
        ...
```

每个Block中包含Attention模块：
```python
class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        ...
        self.attn = Attention(...)
        ...
```

**Hook机制的技术挑战：**
1. **模块路径复杂**：深度嵌套的模块结构导致准确定位目标模块困难
2. **批量注册繁琐**：Transformer中每层都包含attention map，逐个注册hook效率低下

**AnyCapture的技术优势：**

基于上述技术分析，AnyCapture提供了一种更为简洁高效的解决方案，具备以下核心特性：
* 🎯 **精准定位**：支持按变量名精确捕获模型中间结果
* ⚡ **多变量支持**：装饰器支持同时捕获多个目标变量
* 🚀 **高效便捷**：可批量获取Transformer模型中所有层的attention map
* 🔄 **非侵入式设计**：无需修改现有函数代码
* 🎯 **开发友好**：可视化分析完成后无需修改训练代码

## 安装指南
使用pip安装AnyCapture：

```shell
pip install AnyCapture
```

## 使用指南

安装完成后，通过`get_local`装饰器可以便捷地捕获函数内部的局部变量。

### 基础用法：单变量捕获

以捕获`attention_map`变量为例：

**步骤1：在模型文件中添加装饰器**
```python
from anycapture import get_local

@get_local('attention_map')
def your_attention_function(*args, **kwargs):
    ...
    attention_map = ... 
    ...
    return ...
```

**步骤2：在分析代码中激活装饰器并获取结果**
```python
from anycapture import get_local

get_local.activate()  # 激活装饰器
from ... import model  # 注意：模型导入必须在装饰器激活之后

# 加载模型和数据
...
output = model(data)

# 获取捕获的变量
cache = get_local.cache  # 输出格式：{'your_attention_function.attention_map': [attention_map]}
```

捕获结果以字典形式存储在`get_local.cache`中，键值格式为`函数名.变量名`，对应值为变量值列表。

### 基本功能

```python
# 查看缓存内容
print(get_local.cache)

# 清空缓存
get_local.clear()

# 队列功能：限制缓存大小
get_local.activate(max_size=10)  # 只保留最近10次结果
get_local.set_size(5)  # 动态调整为5个元素
```

详细文档请参考：[DOC.md](DOC.md) | [demo.ipynb](https://nbviewer.jupyter.org/github/zzaiyan/AnyCapture/blob/main/demo.ipynb) | [更新日志](UPDATE.md)

## 可视化案例

以下展示了使用AnyCapture对Vision Transformer小型模型（vit_small）进行可视化分析的部分结果。完整案例请参考[demo.ipynb](https://nbviewer.jupyter.org/github/zzaiyan/AnyCapture/blob/main/demo.ipynb)。

由于标准Vision Transformer的所有Attention Map均在`Attention.forward`方法中计算，仅需对该方法添加装饰器，即可批量提取模型12层Transformer的全部Attention Map数据。

**单个Attention Head可视化结果：**

![a head](assets/head.png)

**单层全部Attention Heads可视化结果：**

![heads](assets/heads.png)

**网格级别Attention Map可视化：**

![grid2grid](assets/grids.png)

## 版权信息

**原始作者**: [luo3300612](https://github.com/luo3300612)  
**原始项目**: [Visualizer](https://github.com/luo3300612/Visualizer)  
**当前维护者**: [zzaiyan](https://github.com/zzaiyan)

> 本项目基于luo3300612的Visualizer项目进行重构和功能扩展。为避免与PyPI现有软件包的命名冲突，项目重命名为AnyCapture。特此对原作者的卓越贡献表示诚挚感谢。
