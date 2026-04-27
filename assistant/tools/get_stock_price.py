# coding: utf-8

from langchain.tools import tool


@tool
def get_stock_price(stock_code: str) -> str:
    """根据股票代码获取当前股价。
    Args:
        stock_code: 股票名称或代码
    """
    mock_prices = {"小米": "1850.50", "苹果": "10.25", "国贸": "77884"}
    price = mock_prices.get(stock_code, "67834.34")
    return f"股票 {stock_code} 的当前价格为 {price} 元。"


if __name__ == '__main__':
    print(get_stock_price.func("苹果"))

