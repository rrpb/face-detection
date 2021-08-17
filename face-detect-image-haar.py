
import cv2

#loads a pre-trained classifier from the input file
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image="images/messi.png"
#imread returns 2d/3d array depending upon number of color channels present in the image
image = cv2.imread(image)
#convert this to grayscale as opencv face detector expects gray image- https://stackoverflow.com/questions/12752168/why-we-should-use-gray-scale-for-image-processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO] performing face detection...")
"""
scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.
minSize : Minimum possible object size. Objects smaller than that are ignored.
maxSize : Maximum possible object size. Objects larger than that are ignored.
"""
rects = detector.detectMultiScale(gray, scaleFactor=1.20,minNeighbors=15, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} faces detected...".format(len(rects)))

for (x, y, w, h) in rects:
    # draw the face bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)




