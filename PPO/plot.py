import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

with open('./reward.txt', 'r') as f:
    r_list = f.read().split()

# c = defaultdict(list)

# for i in range(len(r_list)):
#     c['reward'].append(r_list[i])

# c = dict(c)
# df = pd.DataFrame.from_dict(c)
# print(df.head())

# df.plot()
index = []
for i in range(len(r_list)):
    if i % 100 == 0:
        index.append(i)

best_flag = 0
better_flag = 0
for i in r_list[-int(len(r_list)/6):]:
    if i == '-5':
        best_flag += 1
        better_flag += 1
    elif i == '-15':
        better_flag += 1
best = best_flag/(len(r_list)/6)
best = round(best, 5)
better = better_flag/(len(r_list)/6)
better = round(better, 5)

print(best, better)
print(len(index))

ax = plt.gca()
ax.set_ylim(-80, 0)
ymajorLocator = MultipleLocator(10)
ax.yaxis.set_major_locator(ymajorLocator)
plt.title(str(best) + '   ' + str(better))

ax.scatter(index, [float(r_list[i]) for i in index], s = 1, alpha=0.3)
plt.savefig('./reward.png')
plt.show()



