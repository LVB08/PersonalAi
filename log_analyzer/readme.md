# 日志分析助手
## 功能说明
> - 一个能在命令行运行，支持多轮对话，拥有记忆功能。
> - 用户可以连续粘贴多段日志，AI能结合上下文分析，并且最终输出标准的JSON报告

## 文件目录
```
log_analyzer/
├── history_version/        # 存放历史版本代码
├── sample/                 # 存放示例脚本
├── log_analyzer_bot.py     # 核心逻辑：日志分析机器人
├── OutputField.py          # 辅助类：定义输出字段格式
├── PromptTemplate.py       # 辅助类：管理Prompt提示词模板
├── requirements.txt        # 环境依赖：Python库清单
└── readme.md               # 项目文档：项目说明文档
```

## 环境&安装命令
> - 环境为 python3.10
> - pip install --upgrade -r requirements.txt