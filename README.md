# MeXEE402_Finals_StephenGabrielAlojado_JairusSunga

<h1 align="center">Computer Vision using OpenCV</h1>
<p align="center"><b>Finals for MexEE 402 - Electives 2
<br> Prepared by: Stephen Gabriel S. Alojado and Jairus C. Sunga</b></p>

## Introduction

### 16 Basic OpenCV
  - OpenCV or the Open spurve Computer Vision Library is one of the most widely used libraries for computer vision tasks. It provides a comprehensive suite of tools for performing operations such as image processing, object detection, and video analysis. With its wide range of functionalities, OpenCV has become an essential resource for researchers, engineers, and developers working in fields that require visual data analysis.

### Demonstrating Morphological Dilation
  - Morphological dilation is a fundamental operation in image processing, particularly in the field of computer vision. It is part of a group of techniques known as morphological operations, which manipulate the structure or shape of objects within an image based on their spatial relationships. These operations are particularly useful for analyzing and processing images in binary or grayscale formats.

### Enhancing Circuit Patterns in PCB Inspection with Morphological Closing
  - In modern electronics, the reliability and functionality of printed circuit boards (PCBs) are crucial for the performance of various electronic devices. PCB inspection is a vital process to ensure the quality and integrity of the circuit patterns, identifying defects such as missing traces, short circuits, and irregularities that can lead to failure.

  - Morphological image processing, which involves operations like dilation and closing, plays a key role in enhancing the quality of PCB images. These operations help highlight relevant features, such as circuit lines, while minimizing noise and unwanted artifacts. In particular, morphological closing is an essential technique that helps smooth the boundaries of the objects in the image, which is beneficial in PCB inspection tasks where precise detection of circuit patterns is critical.

## Abstract
### 16 Basic OpenCV
  - This project demonstrates the application of fundamental image processing techniques using OpenCV, with a focus on transforming and analyzing image data through various operations.
  - The main objective is to showcase key tasks such as changing an image's color profiles, performing edge detection, manipulating images, and detecting specific features like faces and shapes.

### Enhancing Circuit Patterns in PCB Inspection with Morphological Transformation
  - This project focuses on the application of morphological closing to enhance circuit patterns in PCB (Printed Circuit Board) inspection.
  - The main objective is to utilize morphological operations, specifically closing, to improve the quality and accuracy of PCB pattern recognition during automated inspection processes.
  - Morphological closing, which combines dilation followed by erosion, is employed to fill small gaps, eliminate noise, and enhance the clarity of circuit features.
  - The project applies morphological closing to PCB images to improve the visibility of circuit patterns, making it easier to detect potential defects or irregularities.

## Project Methods

### Part 1: 16 Basic OpenCV Projects

### Part 2: Applying Morphological Transformation for Enhancing PCB Traces

## Conclusion

## Additional Materials

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

```python
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
```

- Imports libraries

```python
image = cv2.imread("Images/butterfly.jpeg")
# cv2_imshow(image)
```
- Reads Image

```python
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
# cv2_imshow(canny_image)
```

- Converts the image to grayscale

```python
kernel = np.ones((5,5), np.uint8)
#Dilation
dilate_image = cv2.dilate(canny_image, kernel, iterations=1)
```

- Erosion and Dilation

# cv2_imshow(dilate_image)
#Erosion
# kernel = np.ones((1,1), np.uint8)
erode_image = cv2.erode(dilate_image,kernel, iterations=1)
# cv2_imshow(erode_image)

display = np.hstack((canny_image,dilate_image,erode_image))
cv2_imshow(display)
```



- PCB Dat


## References

### Part 1: 16 Basic OpenCV Projects

[**butterfly: Lesson 1 & 2**](https://www.vecteezy.com/free-photos/real-butterfly)

[**Bear: Lesson 3**](https://www.pinterest.com/pin/784118985094103174/)

[ **shapess: OpenCv Part2. Lesson 1 & 2**](https://medium.com/illumination/play-a-simple-game-that-proves-telepathy-is-real-ff8f864eac93)

[  **group: OpenCV Part2. Lesson 2**](https://www.cleanpng.com/png-people-business-charleston-denver-news-3920171/)

[**Jairus' Photos**](https://www.facebook.com/jairus.sunga23/photos_by)

[**Stephen's Photos**](https://www.facebook.com/stephen.alojado/photos_by)

[**Kathryn's 1st Photo**](https://m.facebook.com/story.php/?story_fbid=3391156477616600&id=498454503553493)

[**Kathryn's 2nd Photo**](https://noelasinasblog.wordpress.com/2023/12/22/16004/)
