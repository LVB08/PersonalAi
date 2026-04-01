# coding: utf-8

import sys

# 打印完整的版本信息字符串
print("Python 版本 (sys.version):")
print(sys.version)

# 打印版本信息元组，便于程序化判断
print("\nPython 版本信息 (sys.version_info):")
print(sys.version_info)

# 示例：获取主版本号
print(f"\n主版本号: {sys.version_info.major}")