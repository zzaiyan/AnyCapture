# AnyCapture 详细文档

## 进阶用法

### 多变量捕获

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

# 队列功能：限制缓存大小，只保留最近的N个变量值
get_local.activate(max_size=10)  # 只保留最近10次的结果

# 动态调整队列大小
get_local.set_size(5)  # 运行时调整为5个元素
```

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

#### `get_local.activate(max_size=None)`
**功能描述：** 激活装饰器捕获功能

**参数说明：**
- `max_size` (int, optional): 队列最大容量。若为None或非正数，则为无限容量。当记录的变量即将超过最大容量时，会弹出最早的变量。

**使用说明：** 必须在模型导入前调用

**示例：**
```python
get_local.activate()              # 无限容量
get_local.activate(max_size=10)   # 最多保留10个变量值
get_local.activate(max_size=None) # 无限容量
```

#### `get_local.clear()`
**功能描述：** 清空所有缓存数据

**使用场景：** 多次推理前的内存清理

#### `get_local.deactivate()`
**功能描述：** 取消激活装饰器捕获功能

**使用说明：** 取消激活后，装饰器函数将正常执行但不捕获变量。现有缓存数据将被保留。

**示例：**
```python
get_local.activate()      # 激活
# ... 使用装饰器函数 ...
get_local.deactivate()    # 取消激活
# ... 函数正常执行但不捕获变量 ...
get_local.activate()      # 可重新激活
```

#### `get_local.set_size(max_size)`
**功能描述：** 动态设置队列最大容量

**参数说明：**
- `max_size` (int, optional): 队列最大容量。若为None或非正数，则为无限容量。

**返回值：** 规范化后的max_size值（int或None）

**示例：**
```python
get_local.set_size(5)     # 设置容量为5，返回5
get_local.set_size(0)     # 设置为无限容量，返回None  
get_local.set_size("abc") # 无效输入，返回None
```

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

## 技术参考
* [Visualizer by luo3300612](https://github.com/luo3300612/Visualizer)
* [bytecode](https://blog.csdn.net/qfcy_/article/details/118890362)
* [local track1](https://stackoverflow.com/questions/52313851/how-can-i-track-the-values-of-a-local-variable-in-python)
* [local track2](https://stackoverflow.com/questions/19326004/access-a-function-variable-outside-the-function-without-using-global)
* [decorator1](https://stackoverflow.com/questions/1367514/how-to-decorate-a-method-inside-a-class)
* [decorator2](https://stackoverflow.com/questions/6676015/class-decorators-vs-function-decorators)