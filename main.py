#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-19 15:17:49
# @Author  : Sundae (chxhyfx@163.com)
# @Link    : https://github.com/SundaeCHX


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

# 时间及采样点初始化
NUM_POINTS = 400
NUM_TIME = 40
time = np.linspace(0, NUM_TIME, NUM_POINTS)
pre_time = np.linspace(0, 50, 500)


def setRealpos():
    '''
    function:
    ---------
    模拟生成真实值序列.

    returns:
    --------
    @realpos: 真实位置序列. 
    '''
    interval0 = [1 if (i < 10) else 0 for i in pre_time]
    interval2 = [1 if (i >= 10) else 0 for i in pre_time]
    realpos = 5*np.sin(pre_time/3)*interval0+5 * \
        np.sin((pre_time-11)/4)*interval2
    for i in range(92, 235):
        realpos[i] = 0
    for i in range(len(realpos)):
        if realpos[i] > 4.9:
            realpos[i] = 4.9
        if realpos[i] < -4.9:
            realpos[i] = -4.9
    return realpos


def takeMeasurement(realpos):
    '''
    function:
    ---------
    给真实位置值加高斯噪声, 模拟生成一个测量值.

    parameter:
    ----------
    @realpos: float, 某个时刻的真实位置.

    returns:
    --------
    @z: float, 某个时刻的位置测量值. 
    '''
    measurementNoise = 0.25
    z = np.random.normal(realpos, measurementNoise)
    return z


def takeOdometry(realpos_prev, realpos_curr):
    '''
    function:
    ---------
    生成一个上个时刻到当前时刻走过的距离估计值（即速度).

    parameters:
    -----------
    @realpos_prev: float, 前一个时刻位置真实值.
    @realpos_curr: float, 当前时刻位置真实值.

    returns:
    --------
    @u: float, 上个时刻到当前时刻走过的距离估计值
    '''
    processNoise = .05
    if realpos_prev == realpos_curr:
        u = 0
    else:
        u = np.random.normal(realpos_curr - realpos_prev, processNoise)
    return u


def main():
    processNoise = 0.1         # 估计误差，初始化
    measurementNoise = 0.1     # 测量误差，初始化
    estimated_position = []    # list of best-guess estimates
    realpos = setRealpos()     # 真实位置序列
    x = realpos[0]             # 真实位置x，初始化
    p = processNoise           # 方差，根据上个时刻位置估计出的当前位置的方差
    unfilterMeasurements = []  # list of measurements
    measurement_time = []

    for i in range(len(time)):
        # 生成上个时刻到当前时刻这段时间走过的估计距离
        if i == 0:
            u = takeOdometry(realpos[i], realpos[i])
        else:
            u = takeOdometry(realpos[i - 1], realpos[i])
        x += u
        p += processNoise

        # 每1步进行一次测量
        if i % 1 == 0:
            z = takeMeasurement(realpos[i])  # 模拟一次测量
            measurement_time.append(time[i])
            unfilterMeasurements.append(z)

            # 根据测量值修正位置估计值
            y = z - x
            k = p / (p + measurementNoise)
            x = x + k * y
            p = (1 - k)*p

        # 记录修正后的位置估计值
        estimated_position.append(x)

    # 对比预测值和真实值之间的误差
    e = 0
    for i in range(400):
        e = e + abs(estimated_position[i] - realpos[i])
    e = e / 400
    print(e)

    e = 0
    for i in range(350, 400):
        e = e + abs(estimated_position[i] - realpos[i])
    e = e / 50
    print(e)

    # 真实值、模拟测量值、卡尔曼滤波预测值数据可视化
    plt.plot(measurement_time, unfilterMeasurements, color='r')
    plt.plot(time, estimated_position, color='g')
    plt.plot(time, realpos[:400], color='b')
    plt.title('Kalman filtering')
    plt.xlim(0, 40)

    # ARIMA模型预测后续轨迹
    dta = np.array(estimated_position[:400], dtype=np.float)
    dta = pd.Series(dta)
    dta.index = pd.Index(range(400))
    arma_mod = sm.tsa.ARMA(dta, (9, 1)).fit()
    predict_sunspots = arma_mod.predict(399, 500, dynamic=True)

    # ARIMA预测数据可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = dta.ix[0:].plot(ax=ax, color='green')
    predict_sunspots.plot(ax=ax, color='orange')
    plt.title('ARIMA forecast')
    plt.plot(np.linspace(0, 500, 500), realpos, color='blue')
    plt.show()


if __name__ == '__main__':
    main()
