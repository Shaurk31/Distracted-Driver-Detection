# imports
import cv2
import numpy as np
from collections import deque
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt


# video and classifier paths
VIDEO_PATH=('C:/Users/shaur/Desktop/test clips/final/STRAIGHT_DRIVER.mov')

CLASSIFIER_PATH=('C:/Users/shaur/Desktop/test clips/cars_good.xml')

DEBUG_MASK = 0
DEBUG_LINE_CALCS = 0
DEBUG_CAR = 0
DEBUG_LANE = 0
DEBUG_LANE_PCT = 0
DEBUG_CAR_PCT = 0


# Initialize globals
# Car detection Frame of Interest Coordinates for Car and Lane
# Car
roi_car_bot_left = 0
roi_car_bot_right = 0
roi_car_top_left = 0
roi_car_top_right = 0

# Lane
roi_lane_bot_left = 0
roi_lane_bot_right = 0
roi_lane_top_left = 0
roi_lane_top_right = 0

# Previous Frame Lane calculations
prev_value_avail=False
prev_left_slope = 0.0
prev_left_intercept = 0.0
prev_right_slope = 0.0
prev_right_intercept = 0.0


def init_roi_params(image):

    global roi_car_top_right,roi_car_top_left,roi_car_bot_left,roi_car_bot_right
    global roi_lane_top_left,roi_lane_top_right,roi_lane_bot_left,roi_lane_bot_right
    height = image.shape[0]
    width = image.shape[1]

    #Lane Parameters
    roi_lane_bot_left = (width // 5, height)
    roi_lane_bot_right = (width, height)
    roi_lane_top_left = (width * 2 // 5, height * 66 // 100)
    roi_lane_top_right= (width * 17 // 20, height * 66 // 100)

    #Car Parameters
    roi_car_bot_left = (width * 1 // 2, height * 8 // 10)
    roi_car_bot_right = (width * 3 // 4, height * 8 // 10)
    roi_car_top_left = (width * 1 // 2, height * 5 // 10)
    roi_car_top_right = (width * 3 // 4, height * 5 // 10)




#Takes Image and co-ordinates of a Polygon
#returns a masked image

def region_of_interest(image,top_left,top_right,bot_left,bot_right):
    # Make the polygon and initialize mask
    polygons = np.array([[bot_left, bot_right, top_right ,  top_left]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    # Overlay the mask and extract image
    masked_image = cv2.bitwise_and(image, mask, mask=mask)

    if DEBUG_MASK:
        display_and_wait(mask)
        display_and_wait(masked_image)

    return masked_image

# Takes cars detected and eliminates anything
# that's outside our region of interest
# from multiple squares in our region of interest, find the biggest one
def cars_overlap_roi(image,cars):

    height = image.shape[0]
    width = image.shape[1]

    ret_car_arr = []

    # Anchoring the co-ordinates for region of interest
    roi_left_edge_x = roi_car_top_left[0]
    roi_right_edge_x = roi_car_top_right[0]
    roi_top_edge_y = roi_car_top_right[1]
    roi_bot_edge_y = roi_car_bot_right[1]

    for (x, y, w, h) in cars:
        left_edge_x = x
        right_edge_x = x+w
        top_edge_y = y
        bot_edge_y = y+h

        if ((left_edge_x > roi_left_edge_x) and  (left_edge_x < roi_right_edge_x) and
            (top_edge_y > roi_top_edge_y) and (top_edge_y < roi_bot_edge_y) and
            (right_edge_x > roi_left_edge_x) and (right_edge_x < roi_right_edge_x) and
            (bot_edge_y < roi_bot_edge_y) and (bot_edge_y > roi_top_edge_y)
            ):
                ret_car_arr.append((x, y, w, h))
        else:
            if DEBUG_CAR:
                print("Eliminated car")

    bal = 0
    x_big = 0
    y_big = 0
    w_big = 0
    h_big = 0
    for (x, y, w, h) in  ret_car_arr:
        if w*h > bal:
            bal=w*h
            x_big=x
            y_big=y
            h_big=h
            w_big=w
    if bal == 0:
        return[]
    else:
        return ([(x_big, y_big, w_big, h_big)])


# Detects the edges using canny edge detecting algorithm
# input: image(frame)
# output: image[grayscale>blur(noise reduction)>Canny(edge detection)]
def canny_edge_detector(image):
    # converts input image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduces noise from the input image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Creates edge map of input image

    canny = cv2.Canny(blur, 50, 150)
    return canny


# creates coordinates for lines
# input: image(frames), line parameters
# output: array of coordinates
def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # height
    y2 = int(3*y1/4)
    #y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    if DEBUG_LINE_CALCS:
        print("created coordinates from slope:",slope,
              " and intercept:", intercept)
        print("x1:",x1,",y1:",y1,"x2:",x2,"y2:",y2)
    return np.array([x1, y1, x2, y2])

# Filter out values that don't make sense for
# straight lane
def check_slope_intercept_range(slope,intercept):
    ## Check Slope is acceptable
    if(abs(slope)<0.5):
        return(False)

    ## Check Intercept is acceptable
    ## TBD
    return (True)


def record_prev_line_values(left_line, right_line):
    global prev_left_slope, prev_left_intercept
    global prev_right_slope, prev_right_intercept
    global prev_value_avail

    prev_left_slope, prev_left_intercept = left_line
    prev_right_slope, prev_right_intercept = right_line
    prev_value_avail= True
    if DEBUG_LINE_CALCS:
        print ("Recorded left values: Slope->",prev_left_slope,
               ",Intercept->",prev_left_intercept)
        print ("Recorded right values: Slope->",prev_right_slope,
               ",Intercept->",prev_right_intercept)

# Takes all the lines detected and bins them into left and right
# Averages all left to give left lane co-ordinates and all the right ones to give right lane
# input: image(frame), lines(xy coordinates of the lines in the frame)
# output: left line and right line of the lane
def calculate_slope_intercept(image, lines):
    global prev_left_slope, prev_left_intercept
    global prev_right_slope, prev_right_intercept
    global prev_value_avail
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if check_slope_intercept_range(slope,intercept):
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    # if we find both a left fit and a right fit
    if left_fit and right_fit:
        if (prev_value_avail):
            left_fit.append((prev_left_slope,prev_left_intercept))
            right_fit.append((prev_right_slope,prev_right_intercept))

        if DEBUG_LINE_CALCS:
            for l_iter in left_fit:
                print("Left line, slope:", l_iter[0], "intercept:", l_iter[1])
            for r_iter in right_fit:
                print("Right line, slope:", r_iter[0], "intercept:", r_iter[1])
                
        left_fit_pct = np.percentile(left_fit, q=50, axis=0)
        right_fit_pct = np.percentile(right_fit, q=50, axis=0)


        # Store the values for next time
        record_prev_line_values(left_fit_pct,right_fit_pct)
        return np.array([left_fit_pct, right_fit_pct])
    else:
        return None

# Displays lines
# input: image, lines
# output: line image(contains line overlay)
def display_lines(image,lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for slope, intercept in lines:

            line_coordinates = create_coordinates(image, (slope, intercept))
            cv2.line(line_image, (line_coordinates[0], line_coordinates[1]), (line_coordinates[2], line_coordinates[3]),
                     (255, 0, 0), 10)
    return line_image

# Displays image on screen
def display_and_wait(display_image,wait=0):
    cv2.imshow("results", display_image)
    if cv2.waitKey(wait) == 13:
        return



# capture frames from a video
cap = cv2.VideoCapture(VIDEO_PATH)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

#Init counters
car_detected_ctr = 0
no_car_detected_ctr = 0
lanes_found_ctr = 0
lanes_not_found_ctr = 0
lane_crossing_detected_ctr = 0


# Main loop. Goes through the video frame by frame
left_queue = deque(maxlen=15)
right_queue = deque(maxlen=15)
prev_left_pos = 0
prev_right_pos = 0

while True:
    lane_detected = False
    car_detected = False
    ret, frame = cap.read()

    if ret:
        #initialize parameters for region of interest
        init_roi_params(frame)

        #############################################
        # 1. detect lanes in region of interest
        ##############################################
        lane_frame = frame
        canny_image = canny_edge_detector(lane_frame)
        cropped_image = region_of_interest(canny_image,
                                           roi_lane_top_left,roi_lane_top_right,
                                           roi_lane_bot_left,roi_lane_bot_right)
        if DEBUG_MASK:
            display_and_wait(canny_image)
            display_and_wait(cropped_image)

        lines = cv2.HoughLinesP(cropped_image,
                                rho=2,
                                theta=np.pi / 180,
                                threshold=50,
                                lines=np.array([]),
                                minLineLength=10,
                                maxLineGap=50)

        slope_intercept = []
        if lines is not None:
            slope_intercept = calculate_slope_intercept(lane_frame, lines)

            if slope_intercept is not None:
                lanes_found_ctr += 1
                line_image = display_lines(lane_frame, slope_intercept)
                combo_image = cv2.addWeighted(lane_frame, 0.8, line_image, 1, 1)
                lane_detected = True
            else:
                combo_image = lane_frame
                lanes_not_found_ctr += 1
                if DEBUG_LANE:
                    print("HoughLines Couldn't detect lanes in this frame..skip")
        else:
            combo_image = lane_frame
            lanes_not_found_ctr += 1
            if DEBUG_LANE:
                print("HoughLines No lanes in this frame..skip")

        if DEBUG_LANE_PCT:
            lane_detect_pct = 100.0 * lanes_found_ctr // (lanes_not_found_ctr + lanes_found_ctr)
            print("Pct lines  found:",
                  lane_detect_pct, "Total frames:",(lanes_not_found_ctr + lanes_found_ctr))

        ##############################################
        #2. detect car in region of interest
        ##############################################

        car_frame = frame

        # Convert to grayscale
        gray = cv2.cvtColor(car_frame, cv2.COLOR_BGR2GRAY)

        # Detect Cars
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.01,
                                            minNeighbors=2,minSize=(75,75))

        cars_filtered = cars_overlap_roi(gray,cars)

        if DEBUG_CAR:
            if(len(cars)):
                print("Cars found:",len(cars))
                print("After roi:",len(cars_filtered))
            else:
                print("No cars found")
            grayp = region_of_interest(gray,
                                           roi_car_top_left, roi_car_top_right,
                                           roi_car_bot_left, roi_car_bot_right)
            display_and_wait(grayp)

            for (x, y, w, h) in cars:
                cv2.rectangle(gray,
                              (x, y),
                              (x + w, y + h),
                              (0, 0, 255),
                              2)
            display_and_wait(gray)
        # overlap with region of interest


        if len(cars_filtered) == 0:
            no_car_detected_ctr +=1
        else:
            car_detected_ctr +=1
            car_detected = True
        if DEBUG_CAR_PCT:
            car_detect_pct = 100.0 * car_detected_ctr // (car_detected_ctr + no_car_detected_ctr)
            print("Pct frames with cars:", car_detect_pct,
                  " Total frames:", (car_detected_ctr + no_car_detected_ctr))


        for (x, y, w, h) in cars_filtered:
            cv2.rectangle(combo_image,
                          (x, y),
                          (x + w, y + h),
                          (0, 0, 255),
                          2)


        ##############################################
        #3. intersect detection
        ##############################################

        if car_detected and lane_detected:
            slope_left, intercept_left = slope_intercept[0]
            slope_right, intercept_right = slope_intercept[1]
            x, y, w, h = cars_filtered[0]
            left_x = x
            left_y = y+h
            right_x = x+w
            right_y = y+h

            left_line_x = (left_y-intercept_left)//slope_left
            right_line_x = (right_y - intercept_right) // slope_right

            left_difference = left_line_x-left_x
            right_difference = right_line_x - right_x

            left_queue.append(left_difference)
            right_queue.append(right_difference)

            left_avg = sum(left_queue) // len(left_queue)
            right_avg = sum(right_queue) // len(right_queue)

            # Check if we are outside the lane
            if left_avg > -35: #Car is out on left
                prev_left_pos = 1
            else:
                if(prev_left_pos== 1):
                    # Distracted Driver condition:  Car was previously out and now came back in
                    cv2.rectangle(combo_image, (384, 0), (510, 128), (0, 0, 255), -1)
                prev_left_pos = 0

            if right_avg < 35:  # Car is out on right
                prev_right_pos = 1
            else:
                if (prev_right_pos == 1):
                    # Distracted Driver condition:  Car was previously out and now came back in
                    cv2.rectangle(combo_image, (384, 0), (510, 128), (0, 0, 255), -1)
                prev_right_pos = 0


        display_and_wait(combo_image, 1)
    else:
        print("Done processing video")
        car_detect_pct = 100.0*car_detected_ctr//(car_detected_ctr+no_car_detected_ctr)
        lane_detect_pct = 100.0*lanes_found_ctr//(lanes_not_found_ctr+lanes_found_ctr)
        print("Pct frames with cars:",car_detect_pct,". Pct frames with lanes:",lane_detect_pct)
        print("Total frames:",(car_detected_ctr+no_car_detected_ctr))
        break

# De-allocate any associated memory usage




cv2.destroyAllWindows()
