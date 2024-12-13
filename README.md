# MeXEE402_Finals_StephenGabrielAlojado_JairusSunga

<h1 align="center">Computer Vision using OpenCV</h1>
<p align="center"><b>Finals for MexEE 402 - Electives 2
<br> Prepared by: Stephen Gabriel S. Alojado and Jairus C. Sunga</b></p>

## I. Introduction

### 16 Basic OpenCV
  - OpenCV or the Open spurve Computer Vision Library is one of the most widely used libraries for computer vision tasks. It provides a comprehensive suite of tools for performing operations such as image processing, object detection, and video analysis. With its wide range of functionalities, OpenCV has become an essential resource for researchers, engineers, and developers working in fields that require visual data analysis.

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

### Part 1: 16 Basic OpenCV Projects

### Part 2: Applying Morphological Transformation for Enhancing PCB Traces

## IV. Conclusion

## V. Additional Materials

### 16 Basic OpenCV Projects

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

- Tracks the movement of a ball in a video


**Project 2: Face Recognition**

```python
!pip install face_recognition
%cd face_recognition
```

- Install Face Recognition

```python
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
import cv2

# Creating the encoding profiles
face_1 = face_recognition.load_image_file("Face/Jairus.jpg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file("Face/Stephen.jpg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

face_3 = face_recognition.load_image_file("Face/Kathryn.jpg")
face_3_encoding = face_recognition.face_encodings(face_3)[0]

known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding
]

known_face_names = [
                    "Jairus Sunga",
                    "Stephen Grabirel Alojado",
                    "Kathryn Bernardo"
]
```

- creates encoding profiles for three faces


```python
file_name = "Face/Unknown_Kat.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```

- loads an unknown image, detects faces, and compares the face encodings with known face profiles.




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

