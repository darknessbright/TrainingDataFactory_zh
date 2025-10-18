<div align="center">
<img src='../assets/title_image.png' width="250"/>
</div>

[![TrainingDataFactory_zh](assets/title_image.png)](assets/title_image.png)

<div align="center">
<a href="README.md" style="font-size: 24px">简体中文</a> 
</div>

## 👉🏻大模型微调训练数据工厂👈🏻

<center><h3>TrainingDataFactory_zh：简单快速生成你的大模型微调Q&A训练数据</h3></center>

### 摘要

在使用LlamaFactory进行大模型微调前需准备训练数据，对于长文本数据，需将其按照语义截断，并参照文本内容构建高质量、问答模式（Q&A）的训练数据。

本项目采用预训练的BERT模型进行词嵌入，通过计算句子张量与语块张量的相似度，进行按照语义的语块切分。这样可以在保持句子原文流畅性的同时，根据语义中心的变更进行语块切分。

调用本地部署的ollama接口（本地算例不够的朋友可以使用ollama cloud模型），根据分割的语块，批量生成Q&A训练数据。

后续将使用“增强型文本段落分割技术”进行优化。

**Tips:** 如需更多信息请联系作者。邮箱：darknessbright@126.com</u>。

> [!CAUTION]
> 感谢大家对TrainingDataFactory_zh项目的支持与关注！
> 请注意，目前由本作者直接维护的**官方渠道仅有**: [[https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)](https://github.com/darknessbright/TrainingDataFactory_zh).
> ***其他任何网站或服务均非官方提供***，本人对其内容及安全性、准确性和及时性不作任何担保。

## 📣 更新日志

- `2025/10/18` 大模型微调训练数据工厂（中文版）V25.10.16 发布。
- `2025/10/16` 完成初版调试测试。
