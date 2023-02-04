from functools import total_ordering
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

src = cv2.imread('test.bmp')
cv2.imshow('test',src)
cv2.waitKey()
cv2.destroyAllWindows()

b = src[...,0]
g = src[...,1]
r = src[...,2]
channels = src
colors = ('b', 'g', 'r')


for (ch, color) in zip (channels, colors):
    hist_m = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist_m, color = color)
plt.show()


def rgb2hsv(src):
    height, width, channel = src.shape

    bgr = src.astype(np.float) / 255.0
    b = bgr[...,0]
    g = bgr[...,1]
    r = bgr[...,2]


    
    h = np.zeros((height, width), dtype=np.float)
    s = np.zeros((height, width), dtype=np.float)
    v = np.max(bgr, axis=2)


    for i in range(height): 
        for j in range(width):
            if v[i][j] == 0: 
                h[i][j] = 0 
                s[i][j] = 0 
            else: 
                min_rgb = min(bgr[i][j])
                s[i][j] = 1 - (min_rgb / v[i][j])
                
                if v[i][j] == r[i][j]:
                    h[i][j] = 60 * (g[i][j] - b[i][j]) / (v[i][j] - min_rgb)
                elif v[i][j] == g[i][j]:
                    h[i][j] = 120 + (60 * (b[i][j] - r[i][j])) / (v[i][j] - min_rgb)
                elif v[i][j] == b[i][j]:
                    h[i][j] = 240 + (60 * (r[i][j] - g[i][j])) / (v[i][j] - min_rgb)
                if h[i][j] < 0: 
                    h[i][j] += 360

                h[i][j] = h[i][j]/360

    return h,s,v



h,s,v = rgb2hsv(src)

height, width, channel = src.shape
def Histo_Equol(new_v):

    k = 0
    sum = 0
    h_range = [0] * 256
    sum_of_hist = [0] * 256
    img_pixel = height * width
    
    int_v = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            int_v[i][j] = int(new_v[i][j]*255)

    for i in range(height):
        for j in range(width):
            h_range[int_v[i][j]] = h_range[int_v[i][j]]+1


    for i in range(256):
        sum = sum + h_range[i]
        sum_of_hist[i] = sum

    d_v = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            k = int_v[i][j]
            d_v[i][j] = sum_of_hist[k]*(255.0/img_pixel)

            new_v[i][j] = int(d_v[i][j])

    return new_v

new_v = Histo_Equol(v)

def hsv2rgb(new_h, new_s, new_v,a,b):

            
    if new_s[a][b] == 0.0:
        return new_v[a][b],new_v[a][b],new_v[a][b]
    i = int(new_h[a][b]*6.0)
    f = new_h[a][b]*6.0 - i
    p = new_v[a][b]*(1.0 - new_s[a][b])
    q = new_v[a][b]*(1.0 - new_s[a][b]*f)
    t = new_v[a][b]*(1.0 - new_s[a][b]*(1.0-f))
    i = i%6

    if i == 0:
        return new_v[a][b],t,p
    if i == 1:
        return q,new_v[a][b],p
    if i == 2:
        return p,new_v[a][b],t
    if i == 3:
        return p,q,new_v[a][b]
    if i == 4:
        return t,p,new_v[a][b]
    if i == 5:
        return new_v[a][b],p,q

q = [[0 for j in range(width)] for i in range(height)]
p = [[0 for j in range(width)] for i in range(height)]
for a in range(height):
    for b in range(width):
        new_v[a][b] = new_v[a][b]/255
        q[a][b], p[a][b], new_v[a][b] = hsv2rgb(h, s, new_v,a,b)

        


dst = (np.dstack((new_v, p, q)) * 255).astype(np.uint8)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

channels = cv2.split(dst)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist_m = cv2.calcHist([ch], [0], None, [height], [0, width])
    plt.plot(hist_m, color = color)
plt.show()


plt.figure('src2dst_v')
plt.subplot(1, 2, 1)
plt.hist(src.ravel(), height, [0, width])
plt.subplot(1, 2, 2)
plt.hist(dst.ravel(), height, [0, width])
plt.show()
