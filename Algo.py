# import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def readDataFromFile(file):
    data = []
    for line in file:
        line = line[1:-2]
        li = list(line.split(","))
        item = []
        # [date, open, high, low, close, volume, average, barCount]
        # keep date, close and volume
        for i, x in enumerate(li):
            if i == 0:
                y = x.strip("'")
                dt = y[:8]
                tm = y[-8:]
                item.append(dt)
                # if there's time following date, separate them and store both
                if len(x) == 20:
                    item.append(tm)
            elif i == 4:
                item.append(float(x.strip()))
            elif i == 5:
                item.append(int(x.strip()))
        data.append(item)
    return data

def read(filename):
    f = open(filename, 'r')
    try:
        data = readDataFromFile(f)
    finally:
        f.close()
        return data

dailyFileName = "data/SPY_1 day.txt"
minuteFileName = "data/SPY_3 mins.txt"

dailyData = read(dailyFileName)
minuteData = read(minuteFileName)

dailyDates = [datetime.strptime(item[0], '%Y%m%d') for item in dailyData]
dailyClosePrices = [item[1] for item in dailyData]
dailyVolumes = [item[2] for item in dailyData]

minuteDateTime = [datetime.combine(datetime.strptime(item[0],'%Y%m%d'),
                                   datetime.strptime(item[1],'%H:%M:%S').time()) for item in minuteData]
minuteClosePrices = [item[2] for item in minuteData]
minuteVolumes = [item[3] for item in minuteData]

# plt.plot(dailyDates, dailyClosePrices)
# plt.gcf().autofmt_xdate()
# plt.show()

plt.plot(minuteDateTime, minuteClosePrices)
plt.gcf().autofmt_xdate()
plt.show()