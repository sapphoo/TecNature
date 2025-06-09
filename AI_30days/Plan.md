
# 📅 每日学习计划（详细版）

## 🔹 第1周：AI & 深度学习基础

| 天数 | 学习内容 | 实施建议 | 推荐资料 |
|------|----------|-----------|-----------|
| Day 1 | 什么是 AI/ML/DL？区别、演进、应用场景 | 阅读总结 + 画知识图谱 | - [吴恩达AI课简述](https://www.bilibili.com/video/BV1JE411g7XF) |
| Day 2 | 神经网络原理（感知机、多层感知机） | 看图 + 手工算一遍前向传播 | - [3Blue1Brown 神经网络](https://www.bilibili.com/video/BV164411b7dx) |
| Day 3 | 激活函数/损失函数/优化器（ReLU, CrossEntropy, Adam） | 用 PyTorch 实现逻辑回归 | - [PyTorch 官方教程](https://pytorch.org/tutorials/) |
| Day 4 | Transformer 原理：Self-Attention、位置编码 | 阅读可视化文章 + 画结构图 | - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| Day 5 | Token、Embedding、语言模型演进（GPT/BERT） | 理解嵌入向量空间 + tokenizer | - [Hugging Face 教程](https://huggingface.co/transformers/index.html) |
| Day 6 | 实战：使用 HuggingFace pipeline 做文本生成 | 安装 `transformers`，调用 GPT2 | - [Transformers Quick Tour](https://huggingface.co/docs/transformers/quicktour) |
| Day 7 | 小复盘 + 写总结笔记 | 用图/表整理本周知识 | - Notion / Typora / Obsidian |

---

## 🔹 第2周：大模型实战 + Prompt 工程

| 天数 | 学习内容 | 实施建议 | 推荐资料 |
|------|----------|-----------|-----------|
| Day 8 | LLM 是什么？预训练 → 微调 → 推理流程 | 总结模型生命周期 | - [GPT-4 报告](https://openai.com/research/gpt-4) |
| Day 9 | Prompt Engineering：Zero-shot, Few-shot, CoT | 自己设计几个 prompt 测试不同效果 | - [OpenAI Prompt Guide](https://platform.openai.com/docs/guides/prompt-engineering) |
| Day 10 | OpenAI API 使用（chat/completions） | 注册账号 → 拿 key → 用 Python 调用 | - [OpenAI API 文档](https://platform.openai.com/docs) |
| Day 11 | Hugging Face 模型深入使用（AutoModel） | 用 `AutoModelForCausalLM` 写生成器 | - [HF Transformers 教程](https://huggingface.co/docs/transformers) |
| Day 12 | 文本摘要 + 问答系统构建 | 使用 `summarization` 和 `question-answering` pipeline | - [Papers with Code：QA 项目](https://paperswithcode.com/task/question-answering) |
| Day 13 | 多模态模型应用（Whisper, CLIP） | 用 Whisper 做语音识别（`speech → text`） | - [Whisper 官方 GitHub](https://github.com/openai/whisper) |
| Day 14 | 小项目：构建一个 ChatBot Demo | 用 OpenAI 或 GPT2 + prompt | - [Gradio 官方教程](https://www.gradio.app/get_started) |

---

## 🔹 第3周：LangChain + RAG 应用工程化

| 天数 | 学习内容 | 实施建议 | 推荐资料 |
|------|----------|-----------|-----------|
| Day 15 | LangChain 核心模块（LLM, PromptTemplate, Chain） | 跑基础示例，理解链式调用结构 | - [LangChain 中文文档](https://docs.langchain.com.cn) |
| Day 16 | 文档加载 + 分割 + 向量化（FAISS） | 用本地 txt/pdf 建立知识库 | - [LangChain 文档加载教程](https://docs.langchain.com/docs/modules/data_connection/document_loaders/) |
| Day 17 | 构建 RAG 模型（知识检索 + LLM） | 实现一个问答机器人：用户问→搜索→回答 | - LangChain RAG 示例 |
| Day 18 | Agent 机制（ReAct、工具调用、记忆） | 用 `initialize_agent` 实现多轮对话 | - LangChain Agent 示例 |
| Day 19 | 多模型组合调用（OpenAI + 本地模型） | 尝试 LLaMA 或 Mistral 模型结合 | - [Hugging Face LLM](https://huggingface.co/models) |
| Day 20 | Gradio UI 入门 | 用 `gr.Interface()` 接用户输入 | - [Gradio 文档](https://gradio.app/docs/) |
| Day 21 | 项目中期复盘 + 调整结构 | 梳理数据流、模块关系、未来拓展 | - 用 draw.io 画结构图

---

## 🔹 第4周：综合项目实战 + 上线部署

| 天数 | 学习内容 | 实施建议 | 推荐资料 |
|------|----------|-----------|-----------|
| Day 22 | 项目选题（问答、搜索、摘要、语音助手） | 选你感兴趣的方向，写“功能需求”文档 | - 参考 Papers with Code |
| Day 23 | 模块拆解 + 数据准备 | 明确输入/处理/输出的步骤和数据格式 | - 使用 Notion 做项目管理 |
| Day 24 | 主流程构建（知识检索 + 回答） | 搭建完整业务流程 | - 使用 LangChain Chain 模块 |
| Day 25 | 加入工具调用能力 + 多轮记忆 | Agent + MemoryBuffer | - LangChain Agent API |
| Day 26 | 构建 Gradio UI 界面 | 加 Logo、提示语、样式美化 | - Gradio 自定义组件 |
| Day 27 | 项目测试 + Debug + 优化 | 收集用户反馈、修复错误 | - 使用 Git + Markdown 做记录 |
| Day 28 | 本地部署 or Hugging Face Spaces | 上传项目 → 生成链接 | - [HF Spaces 教程](https://huggingface.co/docs/hub/spaces) |
| Day 29 | 准备项目介绍文档（README） | 模型介绍、功能说明、使用示例 | - 用 Markdown 写清楚 |
| Day 30 | 总结复盘 + 分享 | 发布到博客/B站/知乎 | - 可以录屏演示项目 |

---

# 📚 总结版资源推荐合集

| 类型 | 推荐 | 链接 |
|------|------|------|
| 视频课程 | 吴恩达 ML、fast.ai、Google Crash Course | [吴恩达课](https://www.bilibili.com/video/BV1JE411g7XF) / [fast.ai](https://course.fast.ai) |
| 理论讲解 | Illustrated Transformer | [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) |
| 实战框架 | Hugging Face, LangChain, Gradio | [HF](https://huggingface.co/docs) / [LangChain](https://docs.langchain.com.cn) / [Gradio](https://gradio.app) |
| API 平台 | OpenAI Platform | [https://platform.openai.com](https://platform.openai.com) |
| 项目灵感 | Papers with Code | [https://paperswithcode.com](https://paperswithcode.com) |

---

# 🛠️ 实施建议

### ✅ 每天节奏建议：
- 学习时间：**2~3 小时**
  - 1 小时：理论输入（视频/文档）
  - 1 小时：代码实践
  - 0.5 小时：笔记 + 总结

### ✅ 工具建议：
- 用 **Notion / Obsidian** 做学习笔记
- 用 **GitHub** 管理项目代码
- 用 **Colab / Jupyter** 写实验代码
- 用 **ChatGPT/Kimi** 帮你解释不懂的论文段落或调试代码

### ✅ 产出建议：
- 每周输出一篇学习笔记
- 每月输出一个可运行的项目 Demo（带 README）
