import cv2;

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

video = cv2.VideoCapture(0);

# We need to set resolutions.
# so, convert them from float to integer.

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while True:
    check, frame = video.read();
    faces = face_cascade.detectMultiScale(frame,
                                          scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    cv2.imshow('Face Detector', frame)
    result.write(frame)

    key = cv2.waitKey(1);

    if key == ord('q'):
        break

video.release()
result.release()
cv2.destroyAllWindows()
