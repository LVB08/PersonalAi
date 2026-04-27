# coding: utf-8

from langchain.tools import tool


@tool
def calculate_growth_rate(old_value: float, new_value: float) -> float:
    """计算增长率。"""
    return round(((new_value - old_value) / old_value) * 100, 2)
