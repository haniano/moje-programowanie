import numpy as np
from matplotlib import pyplot as plt

labels = list('ABCDEFGHIJ')
values = np.random.randint(10, 30, 10)
variance = np.random.randint(1, 4, 4)

plt.bar(labels, values, color='purple')
plt.title('Bar chart')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.show()

heights = [10, 20, 15]
bars = ['A_long', 'B_long', 'C_long']
y_pos = range(len(bars))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90)
plt.show()