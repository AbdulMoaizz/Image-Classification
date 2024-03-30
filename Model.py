import matplotlib.pyplot as plt
import cv2

img = cv2.imread('Test Img\\jhon ciena.jpg')
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('opencv\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv\\haarcascades\\haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
(x, y, w, h) = faces[0]
face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
"""plt.imshow(face_img)
plt.axis('off')
plt.show()"""

cv2.destroyAllWindows()
for (x, y, w, h) in faces:
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = face_img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return "No eyes detected"

cropped_image = get_cropped_image_if_2_eyes('Test Img\\jhon ciena.jpg')

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.imshow(roi_color, cmap='gray')
if cropped_image == 'No eyes detected':
    print(cropped_image)
else:
    plt.imshow(cropped_image, cmap='gray')
plt.show()
