# Input a .jpg file
# Output matrix with rgb values of that file

# Good Source for picture analysis:
# https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6


if __name__ == '__main__':
    import imageio.v2 as img2
    import matplotlib.pyplot as plt

    pic = img2.imread(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\test.jpg')
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


    # make pixels either black or white 