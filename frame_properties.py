import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Carlo\Repos\floaty_bubbles\pics\for_report\img.png")
# Greyscaling
# Get the width of the image
image_width = img.shape[1]

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive Thresholding
#adaptive_thresh = cv2.adaptiveThreshold(
#    grey,
#    255,
#    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#    cv2.THRESH_BINARY,
#    11, # prev 11
#    5
#)

# Normal Thresholding
ret, adaptive_thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

# Morphological Operations (Dilation followed by Erosion)
dilated_thresh = cv2.dilate(adaptive_thresh, np.ones((5, 5), np.uint8), iterations=1)
dilated_eroded_thresh = cv2.erode(dilated_thresh, np.ones((15, 15), np.uint8), iterations=1)


# Find contours in the inverse of the morphed thresholded image
contours, _ = cv2.findContours(~dilated_eroded_thresh, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area or intensity
contours = sorted(contours, key=cv2.contourArea, reverse=True)

num_contours_to_draw = min(10, len(contours))

for i in range(num_contours_to_draw):
    contour = contours[i]
    cv2.drawContours(dilated_eroded_thresh, [contour], -1, (0, 255, 0), 5)  

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)

        # Calculate major and minor diameters
        major_diameter = max(ellipse[1])
        minor_diameter = min(ellipse[1])

        # Check if the smallest diameter is at least 1/4 of the 
        # largest diameter
        if minor_diameter >= major_diameter / 4:
            # Check if the major axis is at least 150 pixels and 
            # smaller than half the image
            if major_diameter < image_width/2:
                if major_diameter >= 150:
                    # Draw the ellipse
                    cv2.ellipse(dilated_eroded_thresh, ellipse, (0, 255, 0), 10)

cv2.imshow("dilated_eroded_thresh", dilated_eroded_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows

#cv2.imwrite(r"C:\Users\Carlo\Repos\floaty_bubbles\pics\for_report\dilated_eroded_thresh_contours_ellipses.png", dilated_eroded_thresh)




#cv2.imshow("Thresholded", th)
#cv2.imshow("Adaptive_Thresholded", th)
#cv2.imshow("Dilated", morphed_thresh)
#cv2.drawContours(morphed_thresh, contours, -1, (0, 255, 0), 2)
#cv2.imshow("Contours", morphed_thresh)


#plt.hist(img.flat, bins=100, range=(0,255))
#ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Adaptive Thresholding


#morph_kernel = np.ones((2, 2), np.uint8)
#morphed_thresh = cv2.dilate(th, morph_kernel, iterations=1)
#morphed_thresh = cv2.erode(morphed_thresh, morph_kernel, iterations=1)

# Find contours in the inverse of the morphed thresholded image
#contours, _ = cv2.findContours(~morphed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#plt.show()
#cv2.imshow("Original", img)