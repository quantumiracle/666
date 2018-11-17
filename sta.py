import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
raw_data='Road-Accident.csv'
df = pd.read_csv('Road-Accident.csv')
print(df.index)

def statis_column(total_rows, col_index, upper_bound):
    colum=[]
    for i in range (total_rows):
        colum.append(df[col_index][i])
#     print(colum)


    # stat_column=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]
    stat_column=[0 for i in range(upper_bound)]
    stat_num=[i for i in range(upper_bound)]
    print(stat_num)
    for i in range (total_rows):
        for j in range (upper_bound):
            if colum[i]==j:
                stat_column[j]+=1
    plt.plot(stat_num,stat_column, label=col_index)
    plt.xlabel('Value')
    plt.ylabel('frequency')
    plt.title(col_index)
    plt.show()


if __name__ == "__main__":
    start=time.time()
    statis_column(285331, 'vehicle_type', 40)  #285331
    end=time.time()
    print('Time: ', start-end)
