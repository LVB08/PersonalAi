import numpy as np

v = np.array([3, 4])  # 数组表示向量
magnitude = np.linalg.norm(v)  # 计算大小
unit_vector = v / magnitude    # 计算单位向量（方向）
angle = np.arctan2(v[1], v[0]) * 180 / np.pi  # 角度（度）

print(f"大小: {magnitude}")
print(f"单位向量 (方向): {unit_vector}")
print(f"与x轴角度: {angle:.1f}°")