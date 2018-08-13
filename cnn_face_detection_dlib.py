# face detection using
# https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c


# import required packages
import cv2
import dlib
import argparse
import time
import os

import predict

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat', help='path to weights file')
args = ap.parse_args()

base_dir = 'extracted'
img = os.path.basename(args.image)
img_name = os.path.splitext(img)[0]
new_dir = os.path.join(base_dir, img_name)
os.makedirs(new_dir, exist_ok=True)

# load input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

start = time.time()
faces_cnn = cnn_face_detector(image, 1)  # apply face detection (cnn)
end = time.time()
print("Execution Time (in seconds) :")
print("CNN : ", format(end - start, '.2f'))

indx = len(faces_cnn)

# loop over detected faces
for face in faces_cnn:
    x = face.rect.left() - 20
    y = face.rect.top() - 40
    w = face.rect.right() - x + 20
    h = face.rect.bottom() - y + 30

    # draw box over face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # crop face and save
    image_name = str(indx) + '.png'
    new_path = os.path.join(new_dir, image_name)
    # print(new_path)
    crop_img = image[y:y + h, x:x + w]
    cv2.imwrite(new_path, crop_img)

    position = predict.find_pose(new_path)
    cv2.putText(image, position,(x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # print(position)

    indx = indx - 1
    # print()

# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()

# close all windows
cv2.destroyAllWindows()
