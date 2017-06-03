# **Finding Lane Lines on the Road**

**Udacity Self Driving Car Nanodegree - Project #1**
Effendi Dufford
2017/6/1

## Project Goal

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image_t1s0]: ./test_images_output/solidWhiteCurve_s0.jpg "solidWhiteCurve: original"
[image_t1s1]: ./test_images_output/solidWhiteCurve_s1.jpg "solidWhiteCurve: gray + enhanced yellow/white lanes"
[image_t1s2]: ./test_images_output/solidWhiteCurve_s2.jpg "solidWhiteCurve: Canny edges"
[image_t1s3]: ./test_images_output/solidWhiteCurve_s3.jpg "solidWhiteCurve: raw left/right Hough lines"
[image_t1s4]: ./test_images_output/solidWhiteCurve_s4.jpg "solidWhiteCurve: overlaid final detected lanes"

[image_t5s0]: ./test_images_output/solidYellowLeft_s0.jpg "solidYellowLeft: original"
[image_t5s1]: ./test_images_output/solidYellowLeft_s1.jpg "solidYellowLeft: gray + enhanced yellow/white lanes"
[image_t5s2]: ./test_images_output/solidYellowLeft_s2.jpg "solidYellowLeft: Canny edges"
[image_t5s3]: ./test_images_output/solidYellowLeft_s3.jpg "solidYellowLeft: raw left/right Hough lines"
[image_t5s4]: ./test_images_output/solidYellowLeft_s4.jpg "solidYellowLeft: overlaid final detected lanes"

[image_c1]: ./test_videos_output/challenge_ex1.jpg "challenge: hough line misdetection"
[image_c2]: ./test_videos_output/challenge_ex2.jpg "challenge: no bump marker detection"


## My Pipeline

See code in "P1.ipynb" cell 5 ("Build a Lane Finding Pipeline" section).

*Example before/after final result:*
![alt text][image_t5s0]
![alt text][image_t5s4]


## Reflection

### 1. Describe your pipeline

My pipeline consists of 4 steps:

1. Filter and enhance image by lane color
2. Canny edge detection with Gaussian blur and region mask
3. Raw line detection by Hough transform and classify left/right by angle
4. Set left/right lanes by weighted linear polyfit of raw lines


#### Step 1 - Filter and enhance image by lane color

**Function:**
image_s1 = filter_lane_color(image)

**Inputs:**
"image" = Original image of road

**Outputs:**
"image_s1" = Image of road converted to grayscale with enhanced yellow/white areas

**Explanation:**

To start identifying lane lines, the image is first converted from RGB to HSV color space by the **cv2.cvtColor()** function.  Using hue/saturation/value is easier to isolate yellow and white with color masks than combinations of red/green/blue values.  The color masks are created by the **cv2.inRange()** function.

After making yellow and white color masks, they are combined to a single mask.  The original image is then converted to Grayscale with the **grayscale()** helper function, and then combined with the mask using the **cv2.bitwise_or()** function to enhance the masked yellow/white areas (boost value to 255) while still keeping the original image information outside the mask in case the color filters miss some relevant edges.

*Before:*
![alt text][image_t5s0]

*After:*
![alt text][image_t5s1]


#### Step 2 - Canny edge detection with Gaussian blur and region mask

**Function:**
image_s2 = detect_lane_edges(image_s1)

**Inputs:**
"image_s1" = Image of road converted to grayscale with enhanced yellow/white areas

**Outputs:**
"image_s2" = Blank image of detected edges

**Explanation:**

To detect edges, a Gaussian blur is applied to the grayscale enhanced image to smooth out noise by the **gaussian_blur()** helper function, and then Canny edge detection is applied by the **canny()** helper function.  The base low=100/high=200 thresholds worked well enough because it is operating on the grayscale image with enhanced yellow/white areas that make the edge detection easier.

A fixed trapezoidal region mask is applied to the blank image of detected edges by the **region_of_interest()** helper function to remove any edges outside of the expected road surface region.  I tried making a more complicated algorithm to automatically detect the road surface region by road color detection and color masking, but it ended up with too much interference from similar colors in the background scenery, so I stuck with a simple fixed trapezoid region for this project.

*Before:*
![alt text][image_t5s1]

*After:*
![alt text][image_t5s2]


#### Step 3 - Raw line detection by Hough transform and classify left/right by angle

**Function:**
(image_s3, left_lines, right_lines) = detect_lane_lines(image_s2, image)

**Inputs:**
"image_s2" = Blank image of detected edges
"image" = Original image of road 

**Outputs:**
"image_s3" = Image of road with drawn raw left/right lines
"left_lines" = List of left line endpoint coordinates
"right_lines" = List of right line endpoint coordinates

**Explanation:**

To detect lines from the Canny edges, a Hough transform is applied by the **cv2.HoughLinesP()** function.  The helper function was not used for this in order to output the raw line endpoints for further processing.

For **Hough parameters**, the final tuning is to set line judgement threshold = 15, minimum line length = 40 pixels, and the max gap to connect lines = 100 pixels.

The basic thinking is that the detected lines should focus on **tracing the overall lane lines by connecting over multiple lane line markings**, instead of just outlining the individual markings because it's difficult to prevent erroneous short markings from road patches, shadows, etc. and any misdetection of short lines can cause large slope and offset errors while missing the big picture of the actual lane direction.

The Hough parameter tuning method is:

1. Use a relatively long max gap to allow some of the lane lines to connect together to form stronger lines along the driving path, which is important for roads where the lane marking lines are short and spaced out to improve the accuracy of slopes of the detected lines.

2. With the longer lines created by allowing longer gaps, increase the minimum line length to a relatively long value to filter out erroneous short line segments that are created by simply connecting two small edge points with a big gap.

3. Once the max gap and min length are tuned, the line judgement threshold is gradually increased until enough erroneous lines are filtered out while still keeping enough detected lines to maintain some detection of lane location.

**I think it's better to always have some detection lines to work with in the next filtering steps even with some misdetection, so the Hough parameters shouldn't be tuned too tightly or else the lane signal may drop in/out with messier road environments!**

The raw lines are then sorted into left/right groups by checking their angle and horizontal location relative to the center of the image.  Lines with angles that are too close to horizontal (<20 deg) are filtered out.

*Before:*
![alt text][image_t5s2]

*After:*
![alt text][image_t5s3]


#### Step 4 - Set left/right lanes by weighted linear polyfit of raw lines

**Function:**
image_s4 = set_lanes(left_lines, right_lines, image_s3)

**Inputs:**
"image_s3" = Image of road with drawn raw left/right lines
"left_lines" = List of left line endpoint coordinates
"right_lines" = List of right line endpoint coordinates

**Outputs:**
"image_s4" = Image of road with overlaid left/right lanes

**Explanation:**

The final step is to process the groups of raw left/right lines to set the final left and right lanes.  A new helper function **my_linear_polyfit()** is used for each side to combine all of the line x endpoint coordinates, y endpoint coordinates, and calculate the line lengths to group for weighting the linear polyfit using the **np.polyfit()** function.

Since the Hough lines are each represented by just two endpoints, simply applying a linear polyfit without weighting causes the fit line to be easily pulled around by even very short outliers.  To filter this out, weighting the polyfit by line length improves the robustness by keeping the fit closer to the long lines which should be more representative of the actual lane lines due to the Hough parameter tuning method explained in Step 3.

The final left/right lanes are then overlaid on the original road image with the raw Hough lines for reference.

*Before:*
![alt text][image_t5s3]

*After:*
![alt text][image_t5s4]


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming is that the color filtering and edge detection can be thrown off when there are objects (cars, lane guard walls, etc) or discolorations (cement patches, shadows, etc) in the road surface that are similar color to the lane lines.  The edge detection doesn't consider geometery to distinguish types of objects or colorations so it shows up as erroneous lane lines that are hopefully averaged out by the lines detected from the actual lane lines.

*Example of raw line misdetection (pink/green lines):*
![alt text][image_c1]

Another shortcoming is that the line detection doesn't pick up the single lane bump markers in between the lines so the gaps between detected lines can be very long.  Some kinds of roads don't have any lines and just use lots of bumps so this detection method may not work at all on those kinds of roads.

*Example of undetected lane bump marker (lower right side):*
![alt text][image_c2]

Also, this pipeline only processes each image frame independently and doesn't use any temporal information or rolling averages/rate limiting to prevent the detected lane from jumping around at each moment in time.

It also only works for driving straight ahead and can't deal with intersections or sharp turns, or distinguishing types of lane lines such as solid line vs dashed line or double solid lines.


### 3. Suggest possible improvements to your pipeline

Some possible improvements would be to:

1. Add some averaging/rate limiting between multiple frames to prevent the lane line from jumping around unrealistically.

2. Use expected geometery to more accurately define the road surface region and focus line detection to realistic lane markings, such as by perspective correction, using typical car/lane widths, map information, etc.

3. Perform object detection and tracking to avoid line misdetection from occlusion, yellow/white colored objects, etc.

4. Use preceding vehicle tracking as an additional source of information for staying in the lanes (such as in snow or fog when drivers may focus more on following the cars ahead when lane lines aren't visible).
