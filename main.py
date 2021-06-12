#melakukan import library
import numpy as np
import cv2

#membaca gambar
img = cv2.imread('1.png')
shifted = cv2.pyrMeanShiftFiltering(img, 30, 60)

#melakukan konversi ke hitam-putih
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#melakukan binary thresholding dengan Otsu Thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


#mendefinisikan kernel, dan membersihkan noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


# mendefinisikan background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

cv2.imshow('markers', sure_bg)
cv2.waitKey(0)

# mencari sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

cv2.imshow('markers', sure_fg)
cv2.waitKey(0)

# mencari unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# menandai Marker
ret, markers = cv2.connectedComponents(sure_fg)
print(markers)

# menambahkan 1 pada marker agar nilainya menjadi 1 (bukan 0)
markers = markers+1

# Region yang tidak diketahui kita beri nilai 0
markers[unknown==255] = 0

#Melakukan proses watershed atau pengisian air
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,255]
img[markers == 1] = [255,255,255]

filename = "1" + "_output" + ".png"
cv2.imwrite(filename, img)

# menampilakn hasil
cv2.imshow('Hasil', img)
cv2.waitKey(0)