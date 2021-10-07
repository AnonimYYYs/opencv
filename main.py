import cv2
from os import listdir


def viewImage(image_to_show, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    #''' detect faces on pictures
    face_cascade = cv2.CascadeClassifier('C:/PythonProjects/opencv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    for fl in listdir("images"):
        image_path = f"images/{fl}"
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )
        print(f"Faces on picture {fl} detected: " + format(len(faces)))

    #'''
