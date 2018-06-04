<<<<<<< HEAD
#from constants import CASC_PATH, SIZE_FACE, EMOTIONS, SAVE_DATASET_IMAGES_FILENAME, SAVE_DATASET_LABELS_FILENAME, SAVE_DIRECTORY, DATASET_CSV_FILENAME
from constants import *
=======
from constants import CASC_PATH, SIZE_FACE, EMOTIONS, SAVE_DATASET_IMAGES_FILENAME, SAVE_DATASET_LABELS_FILENAME, SAVE_DIRECTORY, DATASET_CSV_FILENAME
>>>>>>> origin
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join


cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
    ] = image
    image = gray_border
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    #  None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d


def flip_image(image):
    return cv2.flip(image, 1)


def data_to_image(data):
    data_image = np.fromstring(
        str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_image(data_image)
    return data_image


data = pd.read_csv(join(SAVE_DIRECTORY, DATASET_CSV_FILENAME))

labels = []
images = []
<<<<<<< HEAD
labels_test = []
images_test = []
index_set = 1
index_test = 1
index = 1
total = data.shape[0]

for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    Usage = row['Usage']
    if image is not None:
        if Usage == "Training":
            labels.append(emotion)
            images.append(image)
            # images.append(flip_image(image))
            index_set += 1
        else:
            labels_test.append(emotion)
            images_test.append(image)
            index_test += 1
    else:
        print("Error")
=======
index = 1
total = data.shape[0]
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    if image is not None:
        labels.append(emotion)
        images.append(image)
        # images.append(flip_image(image))
>>>>>>> origin
    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(images)))
<<<<<<< HEAD
print("image: ", index_set)
print("image_test: ", index_test)

np.save('test_set_fer2013.npy', images_test)        # for validation set
np.save('test_labels_fer2013.npy', labels_test)
=======
>>>>>>> origin
np.save(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME), images)
np.save(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME), labels)
