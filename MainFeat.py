import cv2
import matplotlib.pyplot as plt
import numpy as np

img1=cv2.imread('Images/12.png')
img=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY )
img=cv2.medianBlur(img, 3)
plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.title("Original")

# k-means Segmentation
Z = img.reshape((-1,2))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
ret, res2 = cv2. threshold(res2,127,255,cv2.THRESH_BINARY)
plt.subplot(1,4,2)
plt.imshow(res2,cmap='gray')
plt.title("K-means")

# watershared Segmentation
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img1,markers)
img1[markers == -1] = [255,0,0]

plt.subplot(1,4,3)
plt.imshow(markers,cmap='gray')
plt.title("Watershared")

#thresolding
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.subplot(1,4,4)
plt.imshow(~thresh,cmap='gray')
plt.title("Thresolding")



#Shape
from skimage.measure import label, regionprops
label_img = label(res2)
areas = min([r.area for r in regionprops(label_img)])
centroid = min([r.centroid for r in regionprops(label_img)])

#Texture
from skimage.feature import greycomatrix, greycoprops
g = greycomatrix(img, [1], [0],  symmetric = True, normed = True )
contrast = greycoprops(g, 'contrast')
correlation = greycoprops(g, 'correlation')
energy = greycoprops(g, 'energy')
homogeneity = greycoprops(g, 'homogeneity')

#HOG
img1=cv2.imread('Images/12.png')
from skimage.feature import hog
from skimage.transform import resize
resized_img = resize(img1, (128*4, 64*4))
fh, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(64, 64),cells_per_block=(2, 2), visualize=True, multichannel=True)

ft1=[areas,centroid[0],float(contrast),float(correlation),float(energy),float(homogeneity)]
for i in fh : 
    ft1.append(i)
ft1 = [ '%.2f' % elem for elem in ft1 ]
print(ft1)

#GWT
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
GBW = np.zeros((len(kernels), 2), dtype=np.double)
for k, kernel in enumerate(kernels):
    filtered = ndi.convolve(img, kernel, mode='wrap')
    GBW[k, 0] = filtered.mean()
    GBW[k, 1] = filtered.var()
GBWF=GBW.reshape((-1, 1))
ft2=[areas,centroid[0],float(contrast),float(correlation),float(energy),float(homogeneity)]
for i in GBWF : 
    ft2.append(i)
ft2 = [ '%.2f' % elem for elem in ft2 ]
print(ft2)




