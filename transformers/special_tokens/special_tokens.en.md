# Special Tokens and Chat Templates

[中文](./special_tokens.zh-CN.md)

## What Special Tokens Do

During pretraining, a model learns to predict the next token from a large collection of texts. During post-training for conversation, it learns to answer requests and follow an agreed format for chat and tool use. Special tokens are markers with dedicated token IDs that represent text boundaries, message roles, and control information.

The Llama 3 chat format illustrates these functions.

**Marking text and message boundaries**

- `<|begin_of_text|>` starts the input sequence and usually appears at the beginning of a conversation.
- `<|end_of_text|>` marks the end of a text sequence.
- `<|eot_id|>` ends a message. Another message can follow it in a conversation.

**Identifying message roles**

- `<|start_header_id|>user<|end_header_id|>` introduces content from the user, such as a question.
- `<|start_header_id|>assistant<|end_header_id|>` introduces content from the assistant, such as a generated answer.
- `<|start_header_id|>system<|end_header_id|>` introduces system instructions or background information.

The header boundaries are special tokens. The words user, assistant, and system between them are role names. The message body follows its header and ends with a message boundary marker.

**Organizing tool calls**

Some models use `<tool_call>` and `</tool_call>` around a tool name and its arguments. The model and tokenizer determine the format and token IDs. After the model produces the call, external software executes the tool and adds its result to the conversation as a tool message. The model can then continue its response.

Training establishes the meaning of these markers. Examples associate formatting with message content and response behavior, allowing the model to identify roles and produce the appropriate continuation.

## How Chat Templates Organize Conversations

A conversation is usually stored as a list of messages. Each element has a role identifying its source and content containing its body. One user question and one assistant answer occupy two messages. Two conversational rounds usually occupy four. Tool results can also be messages, while tool calls may use structured fields such as tool_calls.

A chat template converts this list into the model's expected input format. It adds role and boundary markers and joins the message bodies in order. The resulting sequence conveys system instructions, user questions, assistant answers, and tool results.

The template should match the format used to train the model. Different models can use different role markers, ending markers, and arrangements. Using the appropriate template gives the model a familiar conversation format.

`apply_chat_template(..., tokenize=False)` returns formatted text. With `tokenize=True`, it also converts that text into token IDs. The model's embedding layer looks up a vector for each ID and passes the vectors to the Transformer.

## Example One: Encoding and Decoding Plain Text

This example uses the Llama-3.1-8B-Instruct tokenizer to encode a plain string into token IDs and decode them back into text.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text = "what's the wether today in LA?"
encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
print("Decoded with special tokens:", decoded_with_special)

# Decoded with special tokens: <|begin_of_text|>what's the wether today in LA?
```

`AutoTokenizer.from_pretrained` loads the tokenizer. The `text` variable holds the input string.

`tokenizer.encode(text, add_special_tokens=True)` converts the string into integer IDs and adds special tokens according to the tokenizer's rules. The displayed output includes `<|begin_of_text|>` at the start.

`tokenizer.decode(encoded_with_special, skip_special_tokens=False)` converts the IDs back into readable text while displaying special markers. Setting `skip_special_tokens=True` would normally hide those markers in the decoded text while leaving the ID sequence unchanged.

This demonstrates tokenizer encoding and decoding. The `add_special_tokens` option adds markers according to tokenizer rules, while a chat template organizes the complete conversation format.

## Example Two: Formatting a Conversation

The following list contains two conversational rounds. The first discusses today's weather, and the second asks about rain tomorrow. Questions and answers are stored as separate messages.

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

The formatted output is shown below.

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

The `chat` list already contains four messages, including two assistant answers supplied as history. The template formats that existing conversation.

With `tokenize=False`, `encoded_chat` contains the displayed string. With `tokenize=True`, the result would contain token IDs.

The output starts with `<|begin_of_text|>`. The system header introduces system content, user headers introduce questions, and assistant headers introduce answers. Each `<|eot_id|>` closes its message, followed by the next message's header.

The knowledge cutoff and date in this output are default system content inserted by the template. A template can add defaults even when the input list has no explicit system message. The actual date and format depend on the tokenizer template, version, and arguments.

To request an answer to a new question, append its user message to the history and use `add_generation_prompt=True`. In the Llama 3 chat format, this adds an assistant header at the end so the model can begin generating its answer there.

The complete path is a message list formatted by a chat template, formatted text converted into integer token IDs, and token IDs mapped to vectors by an embedding layer. Both special tokens and ordinary text tokens follow this path.
