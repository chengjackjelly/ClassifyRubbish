import cv2
import numpy as np
import random

# 底板图案
bottom_pic = '1.png'
# 上层图案
top_pic = 'baidu000001.jpg'

bottom = cv2.imread(bottom_pic)
top = cv2.imread(top_pic)

h, w, _ = bottom.shape

bottom_high=bottom.shape[0]/5
print(bottom_high)
top_high = top.shape[0]
top_width = top.shape[1]
top_ratio = top_high / top_width
new_width = int(bottom_high)
new_high = int(new_width * top_ratio)
top = cv2.resize(top, (new_width, new_high))
img2=top

#alpha，beta，gamma可调
alpha = 0.7
beta = 1-alpha
gamma = 0

randomx=int((h-new_width)*random.random())
randomy=int((w-new_high)*random.random())
# 权重越大，透明度越低
bottom[randomx:randomx+new_width, randomy:randomy+new_high] = top
#cv2.addWeighted(bottom, 0.5, img2, 0.5, 0)
#overlapping = cv2.addWeighted(bottom,0.8,top,0.2,0)
# 保存叠加后的图片
# cv2.imwrite('overlap(8:2).jpg', overlapping)

cv2.namedWindow('newImage')
cv2.imshow('newImage',bottom)
cv2.waitKey()
cv2.destroyAllWindows()