import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../output/coll_item_item_tuning_2.csv', ',')
'''
d1 = df.loc[df['shr'] == 0]
d2 = df.loc[df['shr'] == 50]
d3 = df.loc[df['shr'] == 100]
d4 = df.loc[df['shr'] == 150]
d5 = df.loc[df['shr'] == 200]
d6 = df.loc[df['shr'] == 250]
d7 = df.loc[df['shr'] == 300]
'''
d8 = df.loc[df['shr'] == 350]
d9 = df.loc[df['shr'] == 400]
d0 = df.loc[df['shr'] == 450]
d7 = df.loc[df['shr'] == 500]


'''
plt.plot(d1['knn'], d1['map'], 'r--')
plt.plot(d2['knn'], d2['map'], 'b--')
plt.plot(d3['knn'], d3['map'], 'r--')
plt.plot(d4['knn'], d4['map'], 'r^')
plt.plot(d5['knn'], d5['map'], 'b^')
plt.plot(d6['knn'], d6['map'], 'g^')
'''
plt.plot(d7['knn'], d7['map'], 'ro')
plt.plot(d8['knn'], d8['map'], 'b^')
plt.plot(d9['knn'], d9['map'], 'go')
temp = plt.plot(d0['knn'], d0['map'], 'bo')

plt.setp(temp, markersize=10)
plt.setp(temp, markerfacecolor='C0')

plt.show()

