from scipy import ndimage
import matplotlib.pyplot as plt
from ccpi.framework import TestData
from ccpi.optimisation.operators import Gradient, FiniteDiff, FiniteDifferenceOperator
import time
import cv2
loader = TestData()

data = loader.load(TestData.CAMERA)

data = TestData.random_noise(data, seed=1)
N = 100
tsx = tsy = 0
tfx = tfy = 0
tcx = tcy = 0
for i in range(100):
    print ("iteration", i)
    t0 = time.time()
    sobel_gx = ndimage.sobel(data.as_array(), axis=0, mode='nearest')
    t1 = time.time()
    tsx += (t1-t0)/N
    sobel_gy = ndimage.sobel(data.as_array(), axis=1, mode='nearest')
    t2 = time.time()
    tsy += (t2-t1)/N

    fd = FiniteDiff(data.geometry, direction=0, bnd_cond='Neumann')
    gx = fd.direct(data)
    t3 = time.time()
    tfx += (t3-t2)/N

    fd.direction=1
    gy = fd.direct(data)
    t4 = time.time()
    tfy += (t4-t3)/N

    t5 = time.time()
    cv_gx = cv2.Sobel(data.as_array(),cv2.CV_64F,0,1,ksize=3)
    t6 = time.time()
    tcx += (t6-t5)/N
    cv_gy = cv2.Sobel(data.as_array(),cv2.CV_64F,1,0,ksize=3)
    t7 = time.time()
    tcy += (t7-t6)/N


print ("SciPy x {} y {}".format(tsx, tsy))
print ("FiniteDiff x {} y {}".format(tfx, tfy))
print ("CV x {} y {}".format(tcx, tcy))

fig = plt.figure()
plt.subplot(3,2,1)
plt.imshow(sobel_gx, cmap='gray')
plt.title('SciPy Sobel')
plt.subplot(3,2,2)
plt.imshow(sobel_gy, cmap='gray')
plt.subplot(3,2,3)
plt.imshow(gx.as_array(), cmap='gray')
plt.title('Finite Diff')
plt.subplot(3,2,4)
plt.imshow(gy.as_array(), cmap='gray')
plt.subplot(3,2,5)
plt.imshow(cv_gx, cmap='gray')
plt.title('OpenCV Sobel')
plt.subplot(3,2,6)
plt.imshow(cv_gy, cmap='gray')
plt.show()