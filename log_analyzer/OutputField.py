# coding: utf-8
"""
@Time: 2026-03-29
@Author: 怀风・Halcyon
@Description: 日志分析助手，输出模板字段定义
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


# 使用pydantic定义输出字段
class LogAnalyzerInfo(BaseModel):
    error_type: str = Field(description="错误类型")
    solution: str = Field(description="解决方案")
    fix_cmd: str = Field(description="用于修复错误的命令")
    serverity: str = Field(description="风险等级评估，high|medium|low")


# 初始化Pydantic输出解析器
cus_parser = PydanticOutputParser(pydantic_object=LogAnalyzerInfo)