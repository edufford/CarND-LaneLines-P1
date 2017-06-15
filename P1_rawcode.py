#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#plt.show

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


import os
os.listdir("test_images/")
print("Current working dir : %s" % os.getcwd())

# Project 1 - Lane Detection Pipeline
#  Effendi Dufford, 2017/6/1
#
# This project detects lane lines in images by applying color/region masks, Canny edge
# detection, Hough transform for determining lines, and setting the left/right lanes by 
# a weighted linear polyfit.  The raw left/right lines and the final detected left/right
# lanes are overlaid on the original image as the output.


def my_lane_detection_pipeline(image, debug_images=False):
    """ Main pipeline to detect lane lines in a road image """

    # Step 1 - Filter and enhance image by lane color
    image_s1 = filter_lane_color(image)
    
    # Step 2 - Canny edge detection with Gaussian blur and region mask
    image_s2 = detect_lane_edges(image_s1)
    
    # Step 3 - Raw line detection by Hough transform and classify left/right by angle
    (image_s3, left_lines, right_lines) = detect_lane_lines(image_s2, image)
    
    # Step 4 - Set left/right lanes by weighted linear polyfit of raw lines
    image_s4 = set_lanes(left_lines, right_lines, image_s3)
    
    # Save images of each step for debugging and documentation
    if debug_images:
        mpimg.imsave('test_images_output/'+image_name.replace('.jpg','_s0.jpg'), image)
        mpimg.imsave('test_images_output/'+image_name.replace('.jpg','_s1.jpg'), image_s1, cmap = 'gray')
        mpimg.imsave('test_images_output/'+image_name.replace('.jpg','_s2.jpg'), image_s2)
        mpimg.imsave('test_images_output/'+image_name.replace('.jpg','_s3.jpg'), image_s3)
        mpimg.imsave('test_images_output/'+image_name.replace('.jpg','_s4.jpg'), image_s4)
    
    # Output image with overlaid raw lane lines and detected left/right lanes
    return image_s4


def filter_lane_color(image):
    """ Filter and enhance image by yellow/white lane colors """
    
    YELLOW_HSV_LOWER = np.array([20, 100, 100])
    YELLOW_HSV_UPPER = np.array([40, 255, 255])
    
    WHITE_HSV_LOWER = np.array([0, 0, 220])
    WHITE_HSV_UPPER = np.array([255, 255, 255])
    
    image_wk = np.copy(image) # working copy
    image_hsv = cv2.cvtColor(image_wk, cv2.COLOR_RGB2HSV) # RGB -> HSV
    
    yellow_mask = cv2.inRange(image_hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    white_mask = cv2.inRange(image_hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    both_mask = yellow_mask + white_mask
    
    image_wk = grayscale(image_wk) # RGB -> GRAY
    image_wk = cv2.bitwise_or(image_wk, both_mask) # GRAY + yellow/white mask
    
    # Output grayscale road image with enhanced yellow/white areas
    return image_wk


def detect_lane_edges(image):
    """ Canny edge detection with Gaussian blur and region mask """
    
    GAUSS_KERNEL = 7 # must be odd <7,5>
    
    CANNY_LOW = 100 # not an edge
    CANNY_HIGH = 200 # definitely an edge
    
    REGION_TRAP_XB = 5 # percent in horizontally from edge for bottom of trapezoid
    REGION_TRAP_XT = 45 # percent in horizontally from edge for top of trapezoid
    REGION_TRAP_YT = 60 # percent down vertically from edge for top of trapezoid
    
    image_wk = np.copy(image) # working copy
    
    # Apply Gaussian blur
    image_wk = gaussian_blur(image_wk, GAUSS_KERNEL)
    
    # Apply Canny edge detection
    image_wk = canny(image_wk, CANNY_LOW, CANNY_HIGH)
    
    # Apply trapezoidal region mask
    im_y = image_wk.shape[0]
    im_x = image_wk.shape[1]
    trap_bl = (np.int32(REGION_TRAP_XB/100*im_x), im_y)
    trap_tl = (np.int32(REGION_TRAP_XT/100*im_x), np.int32(REGION_TRAP_YT/100*im_y))
    trap_tr = (im_x - np.int32(REGION_TRAP_XT/100*im_x), np.int32(REGION_TRAP_YT/100*im_y))
    trap_br = (im_x - np.int32(REGION_TRAP_XB/100*im_x), im_y)
    vertices = np.array([[trap_bl, trap_tl, trap_tr, trap_br]], dtype=np.int32)
    image_wk = region_of_interest(image_wk, vertices)
    
    # Output edge-detected image masked by trapezoidal region
    return image_wk


def detect_lane_lines(image_edges, image_orig):
    """ Raw line detection by Hough transform and classify left/right by angle """
    
    HOUGH_RHO = 1 # distance resolution in pixels of the Hough grid
    HOUGH_THETA = np.pi/180 # angular resolution in radians of the Hough grid
    HOUGH_THRESH = 15 # minimum number of votes (intersections in Hough grid cell) <15,20>
    HOUGH_MIN_LEN = 40 # minimum number of pixels making up a line <40,100>
    HOUGH_MAX_GAP = 100 # maximum gap in pixels between connectable line segments <100,250>
    
    LINE_MIN_ANGLE = 20 # degrees
    
    image_wk = np.copy(image_orig) # working copy
    
    # Run Hough transform on edge-detected image
    raw_lines = cv2.HoughLinesP(image_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH, np.array([]),
                                 HOUGH_MIN_LEN, HOUGH_MAX_GAP)
    
    # Group lines by left/right angle and side of center line
    left_lines = []
    right_lines = []
    x_center = np.int32((image_wk.shape[1]/2))
    for line in raw_lines:
        for x1, y1, x2, y2 in line:
            theta = np.arctan((y2-y1)/(x2-x1)) /np.pi*180
            
            if (theta < -LINE_MIN_ANGLE) and (x1 < x_center) and (x2 < x_center):
                left_lines.append(line)
                
            elif (theta > LINE_MIN_ANGLE) and (x1 > x_center) and (x2 > x_center):
                right_lines.append(line)
    
    # Draw raw left/right lines on road image
    draw_lines(image_wk, left_lines, (255,0,255), 2)
    draw_lines(image_wk, right_lines, (0,255,0), 2)
    
    # Output road image with drawn raw lines and lists of left/right line coordinates
    return (image_wk, left_lines, right_lines)


def set_lanes(left_lines, right_lines, image):
    """ Set left/right lanes by weighted linear polyfit of raw lines """
    
    Y_LANE_EXTRAP = 35 # percent up from bottom of image to extrapolate lane lines
    
    image_wk = np.copy(image) # working copy
    image_lines = np.copy(image_wk)*0 # create a blank to draw lines on
    im_y = image_wk.shape[0]
    
    y1_lane = im_y
    y2_lane = np.int32(im_y - (Y_LANE_EXTRAP/100*im_y))
    
    # Process left lane
    if left_lines:
        z_left = my_linear_polyfit(left_lines)
        x1_lane = np.int32( (y1_lane - z_left[1]) / z_left[0] ) # x = (y-b)/m
        x2_lane = np.int32( (y2_lane - z_left[1]) / z_left[0] )
        
        # Draw left lane on blank image
        cv2.line(image_lines, (x1_lane, y1_lane), (x2_lane, y2_lane), (100,100,100), 15)
    
    # Process right lane
    if right_lines:
        z_right = my_linear_polyfit(right_lines)
        x1_lane = np.int32( (y1_lane - z_right[1]) / z_right[0] ) # x = (y-b)/m
        x2_lane = np.int32( (y2_lane - z_right[1]) / z_right[0] )
        
        # Draw right lane on blank image
        cv2.line(image_lines, (x1_lane, y1_lane), (x2_lane, y2_lane), (100,100,100), 15)
    
    # Overlay detected left/right lanes on road image
    image_wk = weighted_img(image_lines, image_wk)
    
    # Output road image with overlaid left/right lanes
    return image_wk


def my_linear_polyfit(raw_lines):
    """ Apply a linear polyfit to a set of raw line endpoints with weighting by line length """
    
    x = []
    y = []
    weight = []
    
    # Build arrays of all x, y, and weight points
    for line in raw_lines:
        for x1, y1, x2, y2 in line:
            x.extend([x1, x2])
            y.extend([y1, y2])
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            weight.extend([line_length, line_length])
    
    # Apply weighted linear polyfit
    z = np.polyfit(x, y, 1, w=weight)
    
    # Output fit line, z = [m, b]
    return z


""" Process test images """
image_list = os.listdir("test_images/")
for image_name in image_list:
    if image_name.endswith(".jpg"):
        image = mpimg.imread('test_images/' + image_name)
        print('=== Processing', image_name, 'with dimensions', image.shape, '===\n')
        
        # Display base image
        #print('Before processing:')
        #plt.imshow(image)
        #plt.show()
        
        # Process image through pipeline
        image_out = my_lane_detection_pipeline(image, debug_images=True)
        
        # Display processed image
        #print('After processing:')
        #plt.imshow(image_out)
        #plt.show()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    
    result = my_lane_detection_pipeline(image)
    
    return result


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output, audio=False)


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
#%time yellow_clip.write_videofile(yellow_output, audio=False)
yellow_clip.write_videofile(yellow_output, audio=False)


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
#%time challenge_clip.write_videofile(challenge_output, audio=False)
challenge_clip.write_videofile(challenge_output, audio=False)


