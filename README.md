<h1 align="center">Computer Vision using OpenCV</h1>
<p align="center"><b>Final Requirement for MexEE 402 - Electives 2
<br> Prepared by: Stephen Gabriel S. Alojado and Jairus C. Sunga</b></p>

## I. Introduction

### 16 Basic OpenCV
  - OpenCV or the Open sourve Computer Vision Library is one of the most widely used libraries for computer vision tasks. It provides a comprehensive suite of tools for performing operations such as image processing, object detection, and video analysis. With its wide range of functionalities, OpenCV has become an essential resource for researchers, engineers, and developers working in fields that require visual data analysis.

### Demonstrating Morphological Dilation
  - Morphological dilation is a fundamental operation in image processing, particularly in the field of computer vision. It is part of a group of techniques known as morphological operations, which manipulate the structure or shape of objects within an image based on their spatial relationships. These operations are particularly useful for analyzing and processing images in binary or grayscale formats.

### Enhancing Circuit Patterns in PCB Inspection with Morphological Closing
  - In modern electronics, the reliability and functionality of printed circuit boards (PCBs) are crucial for the performance of various electronic devices. PCB inspection is a vital process to ensure the quality and integrity of the circuit patterns, identifying defects such as missing traces, short circuits, and irregularities that can lead to failure.

  - Morphological image processing, which involves operations like dilation and closing, plays a key role in enhancing the quality of PCB images. These operations help highlight relevant features, such as circuit lines, while minimizing noise and unwanted artifacts. In particular, morphological closing is an essential technique that helps smooth the boundaries of the objects in the image, which is beneficial in PCB inspection tasks where precise detection of circuit patterns is critical.

## II. Abstract
### 16 Basic OpenCV
  - This project demonstrates the application of fundamental image processing techniques using OpenCV, with a focus on transforming and analyzing image data through various operations.
  - The main objective is to showcase key tasks such as changing an image's color profiles, performing edge detection, manipulating images, and detecting specific features like faces and shapes.

### Enhancing Circuit Patterns in PCB Inspection with Morphological Transformation
  - This project focuses on the application of morphological closing to enhance circuit patterns in PCB (Printed Circuit Board) inspection.
  - The main objective is to utilize morphological operations, specifically closing, to improve the quality and accuracy of PCB pattern recognition during automated inspection processes.
  - Morphological closing, which combines dilation followed by erosion, is employed to fill small gaps, eliminate noise, and enhance the clarity of circuit features.
  - The project applies morphological closing to PCB images to improve the visibility of circuit patterns, making it easier to detect potential defects or irregularities.

## III. Project Methods

**Data Preparation**
- The data of the PCB Traces was gathered from the kaggle. The proponents search a PCB dataset that is in black in white format for much easier Morphological processing.

**Morphological Dilation**
- This is the tasked assigned to us and we applied it in our dataset. The code below is used to use images with "test" in their names *(those are data with PCB Defects)*.

```python
# Step 1: Specify the folder path
folder_path = "00041"

# Step 2: List and filter files with "test" in their name
all_files = os.listdir(folder_path)
test_files = [file for file in all_files if "test" in file]
test_files = test_files[:5]
```
- We load only ten files for processing with this function `cv2.dilate()`. This will make the dilation process depending on the thresholds, kernels, and iterations that was set.

**Morphological Closing**
- Morphological Closing is done by Erosion then Dilation. This was done by running `cv2.erode()` to the original image and the eroded image was process using the `cv2.dilate()` for the morphological closing operation to be done.

**Region of Interest**
- Due to some of the images needed separate operation of erosion and dilation *(depending on the defect)*, we create a code for selecting specific region of interest *(ROI)* for the morphological transformation operations. This was done through the use of the funciton `cv2.selectROI()`. With the program we did we can select the ROI multiple times before saving the final processed image.

**Applying Morphological Closing and Opening**
- We did a code for applying Morphological Closing and Opening with a feature of selecting the ROI of the image. This helps for the enhancing og the PCB traces and identify the operation if it need closing or opening. The class used was `cv2.MORPH_OPEN` and `cv2.MORPH_CLOSE` under the `cv2.morphologyEx()` module.

## IV. Conclusion

- **Morphological Dilation:**
  - Dilation can be useful for removing short circuits or lines that shouldn't be connected in a PCB image.
  - It works by expanding the foreground objects, which helps in filling small gaps or separating lines that are mistakenly joined.
  - However, dilation alone is not sufficient for closing gaps in PCB traces, as it only widens existing gaps or holes in the image.

- **Limitations of Dilation in PCB Trace Enhancement:**

  - Dilation will not effectively close the gaps in the traces of a PCB where the traces should connect, as it may over-expand and merge disconnected regions.
  - It is better suited for removing noise or addressing undesired connections, such as small short circuits between traces.
  
- **Additional Morphological Operations:**

  - **Erosion:** Erosion reduces the size of the foreground objects. It can be useful for shrinking unwanted blobs or narrow structures that are not part of the PCB trace.
  - **Opening:** This operation involves erosion followed by dilation, and is effective in removing small noise points without affecting larger connected structures. It is especially useful for cleaning up noisy or irrelevant areas in PCB images.
  - **Closing:** A combination of dilation followed by erosion, which is useful for closing small gaps or holes in the PCB traces, ensuring proper connections between trace segments.

- **Selecting Region of Interest (ROI):**

  - ROI selection is crucial in applying the right morphological operation depending on the specific problem being addressed (e.g., removing short circuits, connecting traces, cleaning noise).
  - By carefully selecting the ROI, you can focus the operation on relevant sections of the image, which helps in applying precise transformations without affecting the whole PCB layout.

- **Practical Application in PCB Analysis:**

  - Morphological operations like dilation, erosion, opening, and closing are valuable tools for enhancing PCB trace analysis. These operations can help in detecting issues like missing connections, unwanted short circuits, or trace discontinuities.
  - The choice of operation depends on the problem at hand—whether it's about removing noise, connecting broken traces, or cleaning up the layout.

- **Personal Experience in Image Processing:**

  - Working with morphological operations has provided a great experience in understanding how image processing techniques can be applied programmatically. The ability to manipulate images at a pixel level using code has been an invaluable learning experience in the context of PCB trace enhancement and other practical applications.

## V. Additional Materials

###  Part 1: 16 Basic OpenCV Projects

**Navigating to OpenCV Folder**

```python
%cd OpenCV/
from IPython.display import clear_output
clear_output()
```

**Lesson 1: Changing Image's Color Profiles**

```python
aimport cv2
from google.colab.patches import cv2_imshow

#colorful image - 3 channels
image = cv2.imread("Images/butterfly.jpeg")
print(image.shape)

#grayscale image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(gray.shape)

#HSV Image
HSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
print(HSV.shape)
cv2_imshow(HSV)
```

![Changing Image's Color Profiles](https://github.com/user-attachments/assets/8e8ba0f9-4c01-4d11-91df-e9256228a669)

**Lesson 2: Edge Detection**

- Imports libraries

```python
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
```
- Reads Image

```python
image = cv2.imread("Images/butterfly.jpeg")
# cv2_imshow(image)
```

- Converts the image to grayscale

```python
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
# cv2_imshow(canny_image)
```

- Erosion and Dilation

```python
kernel = np.ones((5,5), np.uint8)
#Dilation
dilate_image = cv2.dilate(canny_image, kernel, iterations=1)
```

- Applies erosion to the dilated image using a kernel,

```python
erode_image = cv2.erode(dilate_image,kernel, iterations=1)
# cv2_imshow(erode_image)
```

- Display

```python
display = np.hstack((canny_image,dilate_image,erode_image))
cv2_imshow(display)
```

![Edge Detection](https://github.com/user-attachments/assets/5bfad25b-ce64-4cec-8387-4f4e666f6d05)


**Lesson 3: Image Manipulation**

- Applies a non-local means denoising filter

```python
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("Images/Bear.jpg")
# cv2_imshow(image)
dst = cv2.fastNlMeansDenoisingColored(image, None, 15, 20, 7, 15)

display = np.hstack((image, dst))
cv2_imshow(display)
```

![Image Manipulation](https://github.com/user-attachments/assets/432f297b-e3ae-46af-a82e-5478670fb032)


**Lesson 4: Drawing Shapes and Writing Text on Images**

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = np.zeros((512, 512, 3), np.uint8)
#uint8: 0 to 255

# Drawing Function
# Draw a Ellipse
cv2.ellipse(img, (256, 256), (100, 50), 45, 0, 360, (255, 0, 255), 3)
# Draw a Polygon
# Define points of the polygon
pts = np.array([[100, 300], [200, 250], [300, 300], [250, 400], [150, 400]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (0, 255, 255))
#Draw a Dashed Line
for i in range(50, 450, 40):
    cv2.line(img, (i, 50), (i + 20, 50), (255, 255, 0), 3)
#Write a Text
cv2.putText(img,"Alojado_Sunga",(12,480),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
# Displaying the Image
cv2_imshow(img)
```

![Drawing Shapes and Writing Text on Images](https://github.com/user-attachments/assets/d22fc04a-b7d9-4792-b8b5-ecb3b5d64be8)



### Part 2: Intermediate Exercises

**Lesson 1: Color Detection**

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
#BGR Image . It is represented in Blue, Green and Red Channels...
image = cv2.imread("Images/shapess.png")
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

# Blue Color
# lower_hue = np.array([65,0,0])
# upper_hue = np.array([110, 255,255])

# Red Color
# lower_hue = np.array([0,0,0])
# upper_hue = np.array([20,255, 255])

# Green Color
# lower_hue = np.array([46,0,0])
# upper_hue = np.array([91,255,255])

# Yellow Color
lower_hue = np.array([21,0,0])
upper_hue = np.array([45,255,255])

mask = cv2.inRange(hsv,lower_hue,upper_hue)
# cv2_imshow(mask)
result = cv2.bitwise_and(image, image, mask = mask)
cv2_imshow(result)
# cv2_imshow(image)
```

![Color Detection](https://github.com/user-attachments/assets/6fd1ab83-02ba-4406-b283-2bec442e9e0b)


**Lesson 2: Face Detection**

```python
import cv2
from google.colab.patches import cv2_imshow

face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")
# img = cv2.imread("images/person.jpg")
img = cv2.imread("Images/group.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)
# print(faces)
for (x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2_imshow(img)
```

![Face Detection](https://github.com/user-attachments/assets/4aaec6e6-e098-4159-8474-dbcda1b37abf)


**Lesson 3: Shape Detection**

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("Images/shapess.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,50,255,1)
contours,h = cv2.findContours(thresh,1,2)
# cv2_imshow(thresh)
for cnt in contours:
  approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
  n = len(approx)
  if n==4:
    # this is a Square
    print("We found a square")
    cv2.drawContours(img,[cnt],0,(255,255,0),3)
  elif n>9:
    # this is a circle
    print("We found a circle")
    cv2.drawContours(img,[cnt],0,(0,255,255),3)
  elif n==3:
    # this is a triangle
    print("We found a triangle")
    cv2.drawContours(img,[cnt],0,(0,255,0),3)
  elif n==6:
    # this is a hexagon
    print("We have a hexagon here")
    cv2.drawContours(img,[cnt],0,255,10)
cv2_imshow(img)
```

![Shape Detection](https://github.com/user-attachments/assets/3ff19693-9843-41c7-b76f-136bbb667c74)


**Part 3: Projects**

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

ball = []
cap = cv2.VideoCapture("videos/video.mp4")
out = cv2.VideoWriter('balloutput.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(1920,1080))
while cap.isOpened():
  ret, frame = cap.read()
  if ret is False:
    break
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  lower_hue = np.array([65,0,0])
  upper_hue = np.array([110, 255,255])
  mask = cv2.inRange(hsv,lower_hue, upper_hue)

  (contours,_)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  center = None

  if len(contours)>0:
    c = max(contours, key=cv2.contourArea)
    ((x,y),radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    try:
      center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
      cv2.circle(frame, center,10, (255,0,0),-1)
      ball.append(center)
    except:
      pass
    if len(ball)>2:
      for i in range(1,len(ball)):
        cv2.line(frame, ball[i-1], ball[i],(0,0,255),5)
  out.write(frame)
out.release()
```



![image](https://github.com/user-attachments/assets/2d803592-0e13-43ec-ab7e-2dc8b939d896)

### Part 2: Applying Morphological Transformation for Enhancing PCB Traces

**Global Processing Morphological Dilation**
```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Specify the folder path
folder_path = "00041"

# Step 2: List and filter files with "test" in their name
all_files = os.listdir(folder_path)
test_files = [file for file in all_files if "test" in file]
test_files = test_files[:5]

# Step 3: Define the kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Step 4: Process each filtered file
for test_file in test_files:
    file_path = os.path.join(folder_path, test_file)
    
    # Read the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {test_file}")
        continue
    
    # Resize the image for manageability
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Apply dilation to enhance the traces
    cleaned_image = cv2.dilate(resized_image, kernel, iterations=2)

    # Step 5: Display the original image and the result after dilation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f"Original Image: {test_file}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_image, cmap='gray')
    plt.title(f"Cleaned Image: {test_file}")
    plt.axis('off')
    
    plt.show()
```
![image](https://github.com/t1pen/MeXEE402_Finals_StephenGabrielAlojado_JairusSunga/blob/main/Revised%20Topic/Outputs/1A.png?raw=true)
![image](https://github.com/user-attachments/assets/edd58779-715f-40ec-a2f1-ad378c47f9ac)
![image](https://github.com/user-attachments/assets/a9f2271b-a01f-44c0-964b-4238dc888f2f)
![image](https://github.com/user-attachments/assets/625aef60-a21c-47c5-8458-f1cf2402bebb)
![image](https://github.com/user-attachments/assets/5750fdbc-630b-42f6-86b1-cbf1c177057d)

**2. Global Processing Morphological Closing**
```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Specify the folder path
folder_path = "00041"

# Step 2: List and filter files with "test" in their name
all_files = os.listdir(folder_path)
test_files = [file for file in all_files if "test" in file]
test_files = test_files[:5]

# Step 3: Define the kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Step 4: Process each filtered file
for test_file in test_files:
    file_path = os.path.join(folder_path, test_file)
    
    # Read the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {test_file}")
        continue
    
    # Resize the image for manageability
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Apply erosion to reduce short circuits
    eroded_image = cv2.erode(resized_image, kernel, iterations=3)

    # Apply dilation to enhance the traces
    cleaned_image = cv2.dilate(eroded_image, kernel, iterations=3)

    # Step 5: Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f"Original: {test_file}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(eroded_image, cmap='gray')
    plt.title("After Erosion")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cleaned_image, cmap='gray')
    plt.title("Cleaned Image")
    plt.axis('off')

    plt.show()
```
![image](https://github.com/user-attachments/assets/d5d14345-1f12-4861-a046-5feb56d7e530)
![image](https://github.com/user-attachments/assets/2add605b-27c1-43aa-91d7-154ffbaf972f)
![image](https://github.com/user-attachments/assets/049f8be6-07ea-4c2d-8c55-f57691cc0e33)
![image](https://github.com/user-attachments/assets/139fb8ca-7602-4cb6-83f3-550003003516)
![image](https://github.com/user-attachments/assets/d20e8c47-d7c2-4752-bbac-e4a0ca8d92df)

**3. Morphological Dilation and Erosion to a Region of Interest**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    """Dummy function for trackbar callbacks."""
    pass

def save_image(image, path):
    """Saves the processed image to a file."""
    cv2.imwrite(path, image)
    print(f"Image saved to {path}")

# Load the image in grayscale
image_path = "00041/00041013_test.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if original_image is None:
    print(f"Error: Image not loaded. Check the file path: {image_path}")
    exit()

# Resize the image for better visualization
resize_width = 600
aspect_ratio = resize_width / original_image.shape[1]
resize_height = int(original_image.shape[0] * aspect_ratio)
processed_image = cv2.resize(original_image, (resize_width, resize_height))

# Create control windows
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 400, 300)  # Set fixed width and height for the Controls window
cv2.namedWindow("Morphological Operations")

# Create trackbars in the "Controls" window
cv2.createTrackbar("Threshold", "Controls", 127, 255, nothing)
cv2.createTrackbar("Kernel Size", "Controls", 3, 20, nothing)
cv2.createTrackbar("Iterations", "Controls", 1, 10, nothing)
cv2.createTrackbar("Operation", "Controls", 0, 1, nothing)  # 0 = Erosion, 1 = Dilation
cv2.createTrackbar("Save", "Controls", 0, 1, nothing)  # 0 = Do not save, 1 = Save

# Allow the user to select and process multiple ROIs
while True:
    # Let the user select ROI from the processed image
    roi = cv2.selectROI("Select ROI", processed_image, showCrosshair=True)
    if roi == (0, 0, 0, 0):  # No ROI selected
        print("No ROI selected. Exiting ROI selection mode.")
        break  # Exit the ROI selection mode

    x, y, w, h = map(int, roi)
    roi_image = processed_image[y:y+h, x:x+w].copy()  # Create a copy of the selected ROI to prevent issues with trackbars

    # Processing loop for this ROI
    while True:
        # Get trackbar positions for settings
        threshold = cv2.getTrackbarPos("Threshold", "Controls")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Controls")
        iterations = cv2.getTrackbarPos("Iterations", "Controls")
        operation = cv2.getTrackbarPos("Operation", "Controls")  # 0 = Erosion, 1 = Dilation
        save_flag = cv2.getTrackbarPos("Save", "Controls")  # Check if Save is triggered

        # Ensure kernel size is odd and not zero
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply binary threshold to the ROI
        _, thresh_image = cv2.threshold(roi_image, threshold, 255, cv2.THRESH_BINARY)

        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion or dilation
        if operation == 0:
            processed_roi = cv2.erode(thresh_image, kernel, iterations=iterations)
        else:
            processed_roi = cv2.dilate(thresh_image, kernel, iterations=iterations)

        # Display the processed image using OpenCV
        display_image = processed_image.copy()
        display_image[y:y+h, x:x+w] = processed_roi  # Update the display image with the processed ROI

        cv2.imshow("Morphological Operations", display_image)

        # Save the image if the Save button (trackbar) is triggered
        if save_flag == 1:
            processed_image[y:y+h, x:x+w] = processed_roi  # Save the changes to the main image
            save_image(processed_image, f"processed_image1.jpg")  # Save with a generic name
            cv2.setTrackbarPos("Save", "Controls", 0)  # Reset the Save button (trackbar) to 0
            break  # Exit the ROI adjustment loop to select another ROI

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC
            break

    if key == 27:  # Exit if ESC is pressed
        break

# Display the original and final processed image using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title("Final Processed Image")
plt.axis('off')

plt.show()

cv2.destroyAllWindows()
```
![image](https://github.com/user-attachments/assets/3aa0b4e6-6c6d-4b8d-88f7-dd53f2c2cfc6)

**4. Morphological Closing and Opening to a Region of Interest**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    """Dummy function for trackbar callbacks."""
    pass

def save_image(image, path):
    """Saves the processed image to a file."""
    cv2.imwrite(path, image)
    print(f"Image saved to {path}")

# Load the image in grayscale
image_path = "00041/00041013_test.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if original_image is None:
    print(f"Error: Image not loaded. Check the file path: {image_path}")
    exit()

# Resize the image for better visualization
resize_width = 600
aspect_ratio = resize_width / original_image.shape[1]
resize_height = int(original_image.shape[0] * aspect_ratio)
processed_image = cv2.resize(original_image, (resize_width, resize_height))

# Create control windows
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 400, 300)  # Set fixed width and height for the Controls window
cv2.namedWindow("Morphological Operations")

# Create trackbars in the "Controls" window
cv2.createTrackbar("Threshold", "Controls", 127, 255, nothing)
cv2.createTrackbar("Kernel Size", "Controls", 3, 20, nothing)
cv2.createTrackbar("Iterations", "Controls", 1, 10, nothing)
cv2.createTrackbar("Operation", "Controls", 0, 1, nothing)  # 0 = Opening, 1 = Closing
cv2.createTrackbar("Save", "Controls", 0, 1, nothing)  # 0 = Do not save, 1 = Save

# Allow the user to select and process multiple ROIs
while True:
    # Let the user select ROI from the processed image
    roi = cv2.selectROI("Select ROI", processed_image, showCrosshair=True)
    if roi == (0, 0, 0, 0):  # No ROI selected
        print("No ROI selected. Exiting ROI selection mode.")
        break  # Exit the ROI selection mode

    x, y, w, h = map(int, roi)
    roi_image = processed_image[y:y+h, x:x+w].copy()  # Create a copy of the selected ROI to prevent issues with trackbars

    # Processing loop for this ROI
    while True:
        # Get trackbar positions for settings
        threshold = cv2.getTrackbarPos("Threshold", "Controls")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Controls")
        iterations = cv2.getTrackbarPos("Iterations", "Controls")
        operation = cv2.getTrackbarPos("Operation", "Controls")  # 0 = Opening, 1 = Closing
        save_flag = cv2.getTrackbarPos("Save", "Controls")  # Check if Save is triggered

        # Ensure kernel size is odd and not zero
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply binary threshold to the ROI
        _, thresh_image = cv2.threshold(roi_image, threshold, 255, cv2.THRESH_BINARY)

        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply opening or closing
        if operation == 0:
            processed_roi = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            processed_roi = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # Display the processed image using OpenCV
        display_image = processed_image.copy()
        display_image[y:y+h, x:x+w] = processed_roi  # Update the display image with the processed ROI

        cv2.imshow("Morphological Operations", display_image)

        # Save the image if the Save button (trackbar) is triggered
        if save_flag == 1:
            processed_image[y:y+h, x:x+w] = processed_roi  # Save the changes to the main image
            save_image(processed_image, f"processed_image.jpg")  # Save with a generic name
            cv2.setTrackbarPos("Save", "Controls", 0)  # Reset the Save button (trackbar) to 0
            break  # Exit the ROI adjustment loop to select another ROI

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC
            break

    if key == 27:  # Exit if ESC is pressed
        break

# Display the original and final processed image using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title("Final Processed Image")
plt.axis('off')

plt.show()

cv2.destroyAllWindows()
```
![image](https://github.com/user-attachments/assets/f12b19ee-137c-4e1f-9f22-c9d588e765a4)

**GUI of the Morphological Transformation in a specific Region of Interest**




## References

### Part 1: 16 Basic OpenCV Projects

- https://www.vecteezy.com/free-photos/real-butterfly
- https://www.pinterest.com/pin/784118985094103174/
- https://medium.com/illumination/play-a-simple-game-that-proves-telepathy-is-real-ff8f864eac93
- https://www.cleanpng.com/png-people-business-charleston-denver-news-3920171/
- https://www.facebook.com/jairus.sunga23/photos_by
- https://www.facebook.com/stephen.alojado/photos_by
- https://m.facebook.com/story.php/?story_fbid=3391156477616600&id=498454503553493
- https://noelasinasblog.wordpress.com/2023/12/22/16004/

### Part 2: Applying Morphological Transformation for Enhancing PCB Traces
- https://www.kaggle.com/datasets/rkuo2000/pcbdata
- https://www.mathworks.com/help/images/morphological-dilation-and-erosion.html

