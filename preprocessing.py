import cv2
from config import IMAGE_SIZE


def preprocess_image(img):

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    normalized = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    return normalized
