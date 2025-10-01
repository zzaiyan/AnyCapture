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

### 高级用法：多变量捕获

AnyCapture支持在单个装饰器中同时捕获多个变量：
```python
from anycapture import get_local

@get_local('attention_map', 'query', 'key', 'value')
def your_attention_function(*args, **kwargs):
    ...
    attention_map = ...
    query = ...
    key = ...
    value = ...
    ...
    return ...
```

**执行和结果获取：**
```python
from anycapture import get_local

get_local.activate()
from ... import model

output = model(data)
cache = get_local.cache 

# 输出示例：
# {
#   'your_attention_function.attention_map': [attention_map],
#   'your_attention_function.query': [query],
#   'your_attention_function.key': [key], 
#   'your_attention_function.value': [value]
# }

# 缓存清理
get_local.clear()
```

### 类方法装饰应用

在PyTorch开发中，通常将模块定义为类，可以直接装饰类内的相关方法：
```python
from anycapture import get_local

class Attention(nn.Module):
    def __init__(self):
        ...
    
    @get_local('attn_map', 'scores')  # 支持同时捕获多个变量
    def forward(self, x):
        ...
        attn_map = ...
        scores = ...
        ...
        return ...
```

### 缓存管理

AnyCapture提供了完善的缓存管理功能：

```python
# 查看当前缓存内容
print(get_local.cache)

# 清空所有缓存数据
get_local.clear()

# 检查缓存状态
print(len(get_local.cache))  # 输出：0
```

详细使用示例请参考[demo.ipynb](https://nbviewer.jupyter.org/github/zzaiyan/AnyCapture/blob/main/demo.ipynb)文件。
## 可视化案例

以下展示了使用AnyCapture对Vision Transformer小型模型（vit_small）进行可视化分析的部分结果。完整案例请参考[demo.ipynb](https://nbviewer.jupyter.org/github/zzaiyan/AnyCapture/blob/main/demo.ipynb)。

由于标准Vision Transformer的所有Attention Map均在`Attention.forward`方法中计算，仅需对该方法添加装饰器，即可批量提取模型12层Transformer的全部Attention Map数据。

**单个Attention Head可视化结果：**

![a head](assets/head.png)

**单层全部Attention Heads可视化结果：**

![heads](assets/heads.png)

**网格级别Attention Map可视化：**

![grid2grid](assets/grids.png)

## 重要事项

### 技术注意点
* **变量覆盖问题**：目标变量在函数内部不应被同名变量覆盖，`get_local`捕获的是变量在函数中的最终值
* **激活时序要求**：执行可视化分析时，必须在模型导入前调用`get_local.activate()`，这是由Python装饰器的导入时执行机制决定的
* **性能影响**：在未调用`get_local.activate()`的情况下，装饰器不会生效，因此对训练性能无任何影响，无需删除装饰器代码
* **内存管理**：进行多次模型推理时，建议使用`get_local.clear()`清空缓存以控制内存占用

## API文档

### 核心接口

#### `get_local(*varnames)`
**功能描述：** 装饰器函数，用于标记需要捕获局部变量的目标函数

**参数说明：**
- `varnames` (str): 目标变量名称，支持多个参数

**返回值：** 装饰后的函数对象

#### `get_local.activate()`
**功能描述：** 激活装饰器捕获功能

**使用说明：** 必须在模型导入前调用

#### `get_local.clear()`
**功能描述：** 清空所有缓存数据

**使用场景：** 多次推理前的内存清理

#### `get_local.cache`
**功能描述：** 缓存字典，存储所有捕获的变量数据

**数据格式：** `{'函数限定名.变量名': [变量值列表]}`

## 应用领域

### 主要应用场景
* **深度学习模型分析**：专业捕获Transformer等模型中的attention maps
* **算法调试优化**：获取函数执行过程中的关键中间变量
* **性能监控分析**：实时监控算法执行中的关键指标变化
* **学术研究开发**：无侵入式提取和分析模型内部计算状态

AnyCapture作为一个通用的局部变量捕获工具，在函数内部变量提取方面具有广泛的应用潜力，可支持各种创新性的应用场景开发。

## 版权信息

**原始作者**: [luo3300612](https://github.com/luo3300612)  
**原始项目**: [Visualizer](https://github.com/luo3300612/Visualizer)  
**当前维护者**: [zzaiyan](https://github.com/zzaiyan)

> 本项目基于luo3300612的Visualizer项目进行重构和功能扩展。为避免与PyPI现有软件包的命名冲突，项目重命名为AnyCapture。特此对原作者的卓越贡献表示诚挚感谢。

## 技术参考
* [Visualizer by luo3300612](https://github.com/luo3300612/Visualizer)
* [bytecode](https://blog.csdn.net/qfcy_/article/details/118890362)
* [local track1](https://stackoverflow.com/questions/52313851/how-can-i-track-the-values-of-a-local-variable-in-python)
* [local track2](https://stackoverflow.com/questions/19326004/access-a-function-variable-outside-the-function-without-using-global)
* [decorator1](https://stackoverflow.com/questions/1367514/how-to-decorate-a-method-inside-a-class)
* [decorator2](https://stackoverflow.com/questions/6676015/class-decorators-vs-function-decorators)
