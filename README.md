# floaty_bubbles

Ellipses are only drawn if:
- it's smallest inner diameter is at least one third of the size of its biggest diameter
- it's smallest diameter is at least 150 pixel long
- it's biggest diameter is not bigger than half the frame width
- when two ellipses are oberlapping: only draw the bigger one

- idea: what if we put in more particles or ignore the top part of the frame?

Main functionality of the code:

Video Input and Output:
The code reads a video file specified by VIDEO_PATH.
It sets up a VideoWriter to create an output video (OUTPUT_VIDEO_NAME) to save the processed frames.
Parameters and Constants:

Several parameters are defined, such as window size (WIN_SIZE), adaptive thresholding block size and constant, morphological kernel size, and more.
Lucas-Kanade Optical Flow:

The code sets up parameters for Lucas-Kanade Optical Flow, although the actual optical flow calculation is not present in the provided code.
Frame Processing Loop:

The code iterates through each frame of the input video.
A region of interest (ROI) is applied to the frame to exclude a specified bottom portion.
Frames are converted to grayscale.
Adaptive Thresholding:

Adaptive thresholding is applied to the grayscale frame using Gaussian mean to dynamically adjust the threshold for different regions.
The resulting binary image highlights regions of interest.
Morphological Operations:

Morphological operations (dilation followed by erosion) are applied to the binary image to enhance and refine the detected regions.
This helps in cleaning up the binary image and preparing it for contour detection.
Contour Detection and Ellipse Fitting:

Contours are found in the inverse of the morphed thresholded image.
The contours are sorted by area, and the top contours are processed.
Ellipses are fitted to the contours using cv2.fitEllipse.
Ellipses that meet certain criteria (size and aspect ratio constraints) are drawn on the frame.
Size Analysis and Visualization:

The code calculates and displays the size of each detected ellipse on the frame.
It calculates the average size of ellipses and the number of ellipses in each frame.
The processed frames are saved to the output video.
Scale Bar:

A scale bar is added to the top-right corner of each frame.
Live Display:

The processed frame is displayed in a window in real-time.
Pressing 'q' quits the display loop.
Diagram Visualization:

After processing all frames, the code plots two diagrams:
Average size of ellipses over time.
Number of ellipses over time.