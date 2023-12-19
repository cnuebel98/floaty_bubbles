import imageio.v2 as img2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
# Good Source for picture analysis:
# https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6
# https://www.mygreatlearning.com/blog/opencv-tutorial-in-python/#:~:text=OpenCV%20is%20a%20Python%20library,%2C%20face%20recognition%2C%20and%20tracking.

def print_info(pic):
    '''prints general information about the given pic in terminal'''
    print('Type of the image : ' , type(pic))
    print('Shape of the image : {}'.format(pic.shape))
    print('Image Hight {}'.format(pic.shape[0]))
    print('Image Width {}'.format(pic.shape[1]))
    print('Dimension of Image {}'.format(pic.ndim))
    print('Image size {}'.format(pic.size))
    print('Maximum RGB value in this image {}'.format(pic.max()))
    print('Minimum RGB value in this image {}'.format(pic.min()))
    
    i=0
    j=0

    # Iterates through every pixel in a given .jpg file
    for row in pic:
        i += 1
        for column in row:
            j += 1

    print("Row count/Height of picture: " + str(i))
    print("Pixel count in picture: " + str(j))
    print("Access a specific pixel: " + str(pic[100, 50]))
    print("Access a specific pixels r value: " + str(pic[100, 50][0]))
    print("Access a specific pixels g value: " + str(pic[100, 50][1]))
    print("Access a specific pixels b value: " + str(pic[100, 50][2]))

if __name__ == '__main__':
    
    pic = img2.imread(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\IMG_5987.png')
    #print_info(pic)
    
    # read image
    flag = cv2.IMREAD_COLOR
    img = cv2.imread(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\IMG_5987.png', flag)
    
    # save image  
    #status = cv2.imwrite(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\IMG_test.png',img)  
    #print("Image written sucess? : ", status)  

    # Access a single pixel
    px = img[100,100]
    print(px)
    
    # modify pixel rgb values
    img[100,100] = [255,255,255]
    print( img[100,100] )

    # access shape of image
    print(img.shape)

    # access size of image
    print(img.size)

    # access data type of image
    print(img.dtype)

    # split and merge rgb channels of image
    b,g,r = cv2.split(img)
    img = cv2.merge((b,g,r))
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    #sets all values in red channel as zero
    img[:,:,2] = 0

    # resize image
    size = [600, 1000]
    img = cv2.resize(img, size)

    # rotate image (Also possible with rotation matrix, see tutorial link at top of the page)
    image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Draw a circle on the image
    center_coordinates = [200, 200]
    radius = 50
    color = [255, 0, 0]
    thickness = 10
    cv2.circle(image, center_coordinates, radius, color, thickness)

    # Draw a rectangle on the image
    start_point = [400, 400]
    end_point = [450, 450]
    cv2.rectangle(image, start_point, end_point, color, thickness)

    # draw a line on the image
    start_point = [10, 200]
    end_point = [10, 500]
    cv2.line(image, start_point, end_point, color, thickness)

    # drawing polylines
    #defining points for polylines
    pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
    is_closed = True
    cv2.polylines(image, [pts], is_closed, color, thickness)

    # write text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,'Tennis',(50, 50), font, 1, color, 5)

    #show image
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()