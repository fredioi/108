# Phase 1 协议修复 - 进度报告

**日期**: 2025-10-17
**状态**: 🟡 部分完成，发现更深层问题

---

## 📋 已完成工作

### 1. ✅ 协议 v2.0.0 设计与实现
- 将 `input_data: Dict[str, Any]` 改为 `input_data_json: str`
- 添加 `protocol_version: str = "2.0.0"` 字段
- 添加 `miner_version: str` 字段
- 提供辅助方法 `get_input_data()` 和 `set_input_data()`
- 保留向后兼容性函数 `synapse_from_v1()`

### 2. ✅ Miner 代码更新
- 使用 `synapse.get_input_data()` 解析 JSON 字符串
- 添加 `synapse.miner_version = "2.0.0"` 响应字段
- 在 blacklist 函数中检查协议版本兼容性

### 3. ✅ Validator 代码
- 无需更改（已使用协议模块的辅助函数）

### 4. ✅ 测试代码更新
- 更新 `test_miner_response.py` 使用新协议

---

## 🔴 发现的核心问题

### 问题：HTTP Header 大小限制

**原始假设**：
- 认为 `Dict[str, Any]` 类型导致序列化失败
- 改成 `str` 类型就能解决

**实际情况**：
```
SynapseParsingError: Could not parse headers into synapse
'total_size': '3594'  # blueprint task
'total_size': '3939'  # characters task
```

**根本原因**：
1. Bittensor 将 **所有 Synapse 字段序列化到 HTTP headers**
2. HTTP headers 有大小限制（通常 4-8KB）
3. 即使 `input_data_json` 是字符串，但内容太大仍然失败

### 为什么数据这么大？

```python
# Characters task 的 input_data:
{
    "user_input": "一个关于AI觉醒的科幻故事",
    "blueprint": {
        "title": "觉醒纪元",
        "genre": "科幻",
        "setting": "2050年的未来世界...",  # 50-200字
        "core_conflict": "...",              # 30-150字
        "themes": [...],
        "tone": "严肃",
        "target_audience": "成人"
    },
    "character_count": 5
}

# JSON 字符串长度: ~1000-1500 字符
# 加上其他字段和 HTTP overhead: 3-4KB
```

---

## 💡 解决方案选项

### 方案 A: 压缩数据（推荐）

```python
import zlib
import base64

class StoryGenerationSynapse(bt.Synapse):
    input_data_compressed: str  # zlib 压缩 + base64 编码

    def set_input_data(self, data: Dict):
        json_str = json.dumps(data, ensure_ascii=False)
        compressed = zlib.compress(json_str.encode())
        self.input_data_compressed = base64.b64encode(compressed).decode()

    def get_input_data(self) -> Dict:
        compressed = base64.b64decode(self.input_data_compressed)
        json_str = zlib.decompress(compressed).decode()
        return json.loads(json_str)
```

**优点**：
- 压缩率 ~60-80%（3KB → 1KB）
- 仍在 HTTP header 限制内
- 透明对 Miner/Validator

**缺点**：
- 增加 CPU 开销（很小）
- 调试时不能直接看到内容

### 方案 B: 只传递关键信息

```python
class StoryGenerationSynapse(bt.Synapse):
    task_type: str
    user_input: str  # 只传用户输入
    context_hash: str  # 完整上下文的哈希

    # blueprint/characters 等大数据存储在：
    # - 链上（Story Registry）
    # - IPFS
    # - Validator 本地缓存
```

**优点**：
- 最小化 header 大小
- 更符合去中心化理念

**缺点**：
- 需要额外的存储层
- 实现复杂度高

### 方案 C: 简化任务类型

只保留 blueprint task，其他任务类型延后：

```python
# Phase 1: 只支持 blueprint
task_types = ["blueprint"]  # input_data 很小

# Phase 2: 添加其他任务
# 使用方案 A 或 B
```

**优点**：
- 快速验证核心功能
- 减少复杂度

**缺点**：
- 功能不完整
- 与原始设计不符

---

## 🎯 推荐行动方案

### 立即实施（1-2天）

1. **实现方案 A（压缩数据）**
   - 修改 `StoryGenerationSynapse` 使用压缩
   - 测试压缩后的大小是否在限制内
   - 验证在测试网工作

2. **添加大小检查**
   ```python
   def validate_size(self):
       # 确保序列化后 < 4KB
       size = len(self.input_data_compressed)
       if size > 4000:
           raise ValueError(f"Data too large: {size} bytes")
   ```

3. **分阶段测试**
   - Step 1: 只测试 blueprint（最小数据）
   - Step 2: 测试 characters（中等数据）
   - Step 3: 测试 story_arc 和 chapters（最大数据）

### 中期考虑（1周内）

如果方案 A 仍然不够：

4. **实现混合方案**
   - Blueprint: 直接传递（小数据）
   - Characters/Story Arc: 压缩传递（中等数据）
   - Chapters: 使用方案 B（链上/IPFS，大数据）

5. **监控与优化**
   - 记录每个任务的数据大小
   - 找出最优压缩参数
   - 实现自适应策略

---

## 📊 当前测试结果

| 任务类型 | 原始大小 | 压缩后大小（预估） | 状态 |
|---------|---------|-------------------|------|
| Blueprint | ~500B | ~200B | ✅ 应该可行 |
| Characters | ~1.5KB | ~600B | ✅ 应该可行 |
| Story Arc | ~2.5KB | ~1KB | ✅ 应该可行 |
| Chapters | ~3.5KB | ~1.4KB | ⚠️ 需要测试 |

---

## 🔄 下一步行动

**今天（2025-10-17）**：
1. [ ] 实现压缩版本的协议 v2.1.0
2. [ ] 更新 Miner/Validator 使用压缩
3. [ ] 本地测试压缩效果
4. [ ] 测试网验证

**明天（2025-10-18）**：
5. [ ] 如果压缩不够，实现方案 B
6. [ ] 压力测试（100+ 并发请求）
7. [ ] 性能基准测试

**本周内**：
8. [ ] 完成 Phase 1 所有任务
9. [ ] 开始 Phase 2（多模型备份）

---

## 💭 经验教训

1. **不要假设问题的根本原因**
   - 我们以为是类型问题，其实是大小问题
   - 应该先读 Bittensor 文档了解限制

2. **测试驱动开发**
   - 应该先测试最小可行版本
   - 然后逐步增加复杂度

3. **阅读其他 Subnet 的代码**
   - 看看他们如何处理大数据
   - 学习最佳实践

---

## 📚 参考资源

- [Bittensor Synapse Documentation](https://docs.bittensor.com)
- [HTTP Header Size Limits](https://stackoverflow.com/questions/686217/maximum-on-http-header-values)
- [zlib Compression in Python](https://docs.python.org/3/library/zlib.html)

---

**结论**: Phase 1 的核心挑战是 HTTP header 大小限制，而不是数据类型。需要实现数据压缩或重新设计协议。
