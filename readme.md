# 大作业1：SFT
## 微调和评估
由于在原notebook文件中修改，所以直接运行project1/finetune-eval.ipynb即可


- 微调后模型文件（basic，微调1）地址：https://jbox.sjtu.edu.cn/l/812DLF

## 其他脚本
project1/scripts目录下放置所有的完成报告和探究所需要的脚本文件，为方便运行，也同样设置为notebook文件
  
- compare.ipynb:对比SFT和PLM模型的不同条目的区别，主要是为了case_study
- test.ipynb:用于测试我们的只计算output的loss方法是否成功，以及一些原模型和微调后模型直接进行推理的实验

----

# 大作业2：聊天机器人
本次作业我们完成了所有3个bonus，主体框架采用lora微调。
文件目录finetune中包含所有的微调文件；scripts目录下包含所有的对话实现文件。

主体对话文件为scripts/conmmunicate.ipynb，将后面所示模型下载并放入路径即可。

## lora微调
- 3B模型微调代码地址如下:https://jbox.sjtu.edu.cn/l/N1iCFD
- 用于对比的0.5B模型使用lora地址如下：https://jbox.sjtu.edu.cn/l/z1hh6L
- 训练代码为finetune_lora.py, 合并模型实现为get_lora.py


## 外部知识库
- 实现见script/try_RAG.ipynb
- 知识库KB包含
  - 人名信息，王者荣耀信息，部分高级知识化学物理等信息
## 虚拟人
- prompt指令实现：见virtual.ipynb
    - 设置了几个不同的角色，完成基础的扮演工作。
- 微调实现：采用甄嬛数据集进行few-shot微调，加载模型后可以直接使用huanhuan.ipynb进行对话。模型地址在：https://jbox.sjtu.edu.cn/l/UHH22m