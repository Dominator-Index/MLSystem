# Special Tokens 与 Chat Template

[English](./special_tokens.en.md)

## Special Tokens 的作用

模型在预训练阶段通过大量文本学习预测下一个 token。在面向对话的后训练阶段，模型学习根据用户请求生成回答，以及按照约定的格式进行聊天和工具调用。Special tokens 是具有专用 token ID 的特殊标记，用来表示文本边界、消息角色和交互中的控制信息。

以 Llama 3 的对话格式为例，可以从下面几类标记理解它们的作用。

**标记文本或消息的开头和结尾**

- `<|begin_of_text|>` 表示整个输入序列的开始，通常放在对话开头。
- `<|end_of_text|>` 表示文本序列的结束。
- `<|eot_id|>` 表示一条消息的结束。在多轮对话中，它后面可以继续接下一条消息。

**区分消息角色**

- `<|start_header_id|>user<|end_header_id|>` 表示接下来的正文来自用户，例如用户提出的问题。
- `<|start_header_id|>assistant<|end_header_id|>` 表示接下来的正文来自助手，例如模型生成的回答。
- `<|start_header_id|>system<|end_header_id|>` 表示接下来的正文是系统消息，例如助手的行为要求或背景信息。

角色头中的 `<|start_header_id|>` 和 `<|end_header_id|>` 是特殊 token，中间的 user、assistant 和 system 是角色名称。消息正文放在角色头之后，再用消息结束标记收尾。

**组织工具调用**

一些模型使用 `<tool_call>` 和 `</tool_call>` 包住工具名称与参数。具体标记以及它们对应的 token ID 由模型和 tokenizer 决定。模型生成调用内容后，外部程序执行工具，再把结果作为工具消息加入对话，供模型继续生成回答。

模型通过训练学习这些标记的含义。训练数据把标记、消息内容和回答方式放在一起，让模型学会识别当前消息的角色以及接下来应生成的内容。

## Chat Template 如何组织对话

多轮对话通常保存为一个消息列表。每个元素是一条消息，通过 role 表示角色，通过 content 保存正文。例如，一次用户提问和助手回答对应两条消息，两轮对话通常对应四条消息。工具返回结果也可以作为消息加入列表，工具调用还可能使用 tool_calls 等结构化字段。

Chat template 是把消息列表转换成模型输入格式的模板。它为每条消息添加角色标记和边界标记，再按顺序拼接消息正文。这样，模型就能够从输入中识别系统要求、用户问题、助手回答和工具结果。

模板需要与模型训练时使用的格式匹配。不同模型可以采用不同的角色标记、结束标记和排列方式。推理时使用相应的模板，能够让模型按照熟悉的格式处理对话。

`apply_chat_template(..., tokenize=False)` 返回拼接后的文本。`apply_chat_template(..., tokenize=True)` 则进一步将文本转换为 token IDs。模型中的 embedding 层再根据这些 ID 取出对应的向量，交给 Transformer 计算。

## 例子一：普通文本的编码与解码

下面使用 Llama-3.1-8B-Instruct 的 tokenizer，把一段普通文本编码成 token IDs，再解码回文本。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text = "what's the wether today in LA?"
encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
print("Decoded with special tokens:", decoded_with_special)

# Decoded with special tokens: <|begin_of_text|>what's the wether today in LA?
```

`AutoTokenizer.from_pretrained` 加载 tokenizer。`text` 保存要处理的字符串。

`tokenizer.encode(text, add_special_tokens=True)` 把字符串转换成整数 ID 序列，并根据 tokenizer 的规则添加特殊 token。在展示的输出中，文本开头加入了 `<|begin_of_text|>`。

`tokenizer.decode(encoded_with_special, skip_special_tokens=False)` 把这些 ID 转回可读文本，同时保留特殊 token 的显示形式，所以打印结果中可以看到 `<|begin_of_text|>`。如果设置 `skip_special_tokens=True`，解码文本通常会隐藏特殊标记，原有 ID 序列保持不变。

这个例子展示了 tokenizer 的编码与解码过程。`add_special_tokens=True` 负责按 tokenizer 的规则添加标记，完整的聊天消息格式则由 chat template 组织。

## 例子二：多轮对话的格式化

下面的消息列表包含两轮对话。第一轮讨论今天的天气，第二轮询问明天是否下雨。用户问题和助手回答分别保存为独立消息。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

chat = [
    {"role": "user", "content": "What's the weather today in LA?"},
    {"role": "assistant", "content": "The weather in LA is sunny with a high of 75°F."},
    {"role": "user", "content": "Will it rain tomorrow?"},
    {"role": "assistant", "content": "No, it's expected to be clear with a low of 58°F."},
]

encoded_chat = tokenizer.apply_chat_template(chat, tokenize=False)
print("Encoded chat:", encoded_chat)
```

格式化后的文本如下。

```text
Encoded chat: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

What's the weather today in LA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The weather in LA is sunny with a high of 75°F.<|eot_id|><|start_header_id|>user<|end_header_id|>

Will it rain tomorrow?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

No, it's expected to be clear with a low of 58°F.<|eot_id|>
```

`chat` 中已经保存了四条消息，包括两条用户消息和两条助手消息。这里的天气回答是预先提供的对话内容。`apply_chat_template` 负责格式化这段历史。

`tokenize=False` 表示结果是字符串，因此 `encoded_chat` 保存的是上面展示的文本。若使用 `tokenize=True`，就会得到 token IDs。

输出从 `<|begin_of_text|>` 开始。随后，system 的角色头引出系统消息，user 的角色头引出用户问题，assistant 的角色头引出助手回答。每条消息末尾的 `<|eot_id|>` 表示该消息结束，下一条消息接着使用自己的角色头。

示例输出中的知识截止时间和日期来自模板补入的默认系统内容。输入列表中没有显式提供 system 消息，模板仍然可以添加默认内容。实际日期和格式取决于 tokenizer 的模板、版本与参数。

如果要让模型回答一个新的问题，可以在历史列表末尾追加 user 消息，再使用 `add_generation_prompt=True`。在 Llama 3 的聊天格式中，这会在末尾添加 assistant 的起始消息头，让模型从这里开始生成回答。

整个处理过程可以概括为消息列表经过 chat template 变成格式化文本，格式化文本经过 tokenizer 变成整数 token IDs，token IDs 再经过 embedding 层变成向量。Special tokens 和普通文本 token 都参与这个过程。
