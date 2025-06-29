## 👋 toyRLVR 是一个使用RLVR训练LLM的个人学习项目

**项目特点：**

- 仅基于pytorch的完全手工实现，不依赖huggingface、verl、ray、vLLM、deepspeed等第三方库，聚焦技术本质
  - 模型：实现标准transformer decoder(MHA, FFN, RMSNorm, position encoding, weight typing, ...)
  - 训练：实现SFT、GRPO RL算法
  - 推理：实现kv cache、贪婪/随机采样、约束解码
- 极小计算资源：运行于单机单卡，基于multiprocessing高效调度CPU和GPU多进程异步进行强化样本生成和训练

**目标任务：**

- 只包含纯数字和`+`、`-`的数学表达式求值，如`8+50-15-34+55`

**效果展示：**

1. **step-1:** 使用SFT训练带有思考过程的样本。**step-2:** 使用RL扩展能力到更长的表达式


```commandline
IN: 36+18+44+24
OUT: <THINK>36+62+24,36+86,122<RES>122<EOS>
```
```commandline
IN: 36+33-33+4+42
OUT: <THINK>36+33-33+46,36+33+13,36+46,82<RES>82<EOS>
```

2. 基于纯RL直接强化未经过任何训练的模型，期望观察到无任何人类数据下模型的思维过程。尚未完成训练