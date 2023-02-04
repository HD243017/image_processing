from functools import total_ordering
from tkinter import *
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import max_error
from sklearn.metrics.pairwise import euclidean_distances

src = cv2.imread('dom4.jpg')
cv2.imshow('test',src)
cv2.waitKey()
cv2.destroyAllWindows()

def distanceRePoint(distance, k):
    min = distance[0]
    minIndex = 0
    for i in range(0,k):
        if distance[i] < min:
            min = distance[i]
            minIndex = i
    return minIndex

def rePointclus(data,k,cdata):
    height, width, channel = data.shape
    b = [0 for j in range(k)]
    g = [0 for j in range(k)]
    r = [0 for j in range(k)]

    avgb = [0 for j in range(k)]
    avgg = [0 for j in range(k)]
    avgr = [0 for j in range(k)]
    count = [0 for j in range(k)]

    for i in range(0,k):
        b[i] = 0
        g[i] = 0
        r[i] = 0
        avgb[i] = 0
        avgg[i] = 0
        avgr[i] = 0
        count[i] = 0

    
    for y in range(0,height):
        for x in range(0, width):
            for j in range(0,k):
                if(cdata[y][x] == j):
                    b[j] += data[y][x][0]
                    g[j] += data[y][x][1]
                    r[j] += data[y][x][2]
                    count[j] += 1

    for i in range(0,k):
        avgb[i] = b[i]/count[i]
        avgg[i] = g[i]/count[i]
        avgr[i] = r[i]/count[i]

    return avgb, avgg, avgr

def kmeans():
    k = 4
    m = 10
    data = np.array(src)

    height, width, channel = data.shape

    cdata = [[0 for j in range(width)] for i in range(height)]

    w = [0 for j in range(k)]
    h = [0 for j in range(k)]
    for i in range(k):
        w[i] = random.randint(0,width)
        h[i] = random.randint(0,height)

        print(w,h)

    distance = [0 for i in range(k)]
    for y in range(0,height):
        for x in range(0,width):
            for j in range(0,k):
                b = abs(int(data[y][x][0])-int(data[h[j]][w[j]][0]))
                g = abs(int(data[y][x][1])-int(data[h[j]][w[j]][1]))
                r = abs(int(data[y][x][2])-int(data[h[j]][w[j]][2]))
                distance[j] = int(math.sqrt(r * r + g * g)) + int(math.sqrt(g * g + b * b)) + int(math.sqrt(r * r + b * b))
            cdata[y][x] = distanceRePoint(distance,k)

    avgb, avgg, avgr = rePointclus(data, k, cdata)

    for z in range(0,m):
        for y in range(0,height):
            for x in range(0,width):
                for j in range(0,k):
                    b = abs(int(data[y][x][0])-int(avgb[j]))
                    g = abs(int(data[y][x][1])-int(avgg[j]))
                    r = abs(int(data[y][x][2])-int(avgr[j]))
                    distance[j] = int(math.sqrt(r * r + g * g)) + int(math.sqrt(g * g + b * b)) + int(math.sqrt(r * r + b * b))
                cdata[y][x] = distanceRePoint(distance,k)
        avgb, avgg, avgr = rePointclus(data, k, cdata)

    for y in range(0,height):
        for x in range(0,width):
            data[y][x][0] = avgb[cdata[y][x]]
            data[y][x][1] = avgg[cdata[y][x]]
            data[y][x][2] = avgr[cdata[y][x]]

    cv2.imshow('dst', data)
    cv2.waitKey()
    cv2.destroyAllWindows()

kmeans()

dds = src.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, bestLabels, centers = cv2.kmeans(dds, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = centers.astype(np.uint8)
dst = centers[bestLabels].reshape(src.shape)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()


    
