import cv2


def viewImage(image_to_show, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    ''' simple image
    
    image = cv2.imread("./images/cat.jpg")
    viewImage(image, "cat")
    '''

    ''' cropping dog image
    
    image = cv2.imread("./images/dog_crop.jpg")
    cropped = image[1:200, 250:550]
    viewImage(cropped, "Dog after cropping")
    '''

    ''' resizing cat image
    
    image = cv2.imread("./images/cat_big.jpg")
    scale_percent = 30  # Это % от изначального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    viewImage(resized, f"{scale_percent}% from original size")
    '''

    ''' flipping dog
    
    image = cv2.imread("./images/dog_flip.jpg")
    (h, w, d) = image.shape
    center = (w // 2, h // 2)
    flip_angle = 180
    M = cv2.getRotationMatrix2D(center, flip_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    viewImage(rotated, f"Flip by {flip_angle} degrees")
    '''

    ''' colored cat to gray-scaled and to black/white
    
    image = cv2.imread("./images/cat_colored.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(image, 127, 255, 0)
    viewImage(gray_image, "Gray-scaled")
    viewImage(threshold_image, "Black/white")
    print(ret)
    '''

    ''' blurring cat
    
    
    image = cv2.imread("./images/cat_clear.jpg")
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    viewImage(blurred, "Blurred image")
    '''

    ''' draw rectangle above cat face

    image = cv2.imread("./images/cat_draw_rectangle.jpg")
    output = image.copy()
    cv2.rectangle(output, (154, 78), (315, 214), (20, 20, 255), 3)
    viewImage(output, "Draw rectangle")
    '''

    ''' split two cats with line

    image = cv2.imread("./images/two_cats.jpg")
    output = image.copy()
    cv2.line(output, (537, 26), (537, 255), (20, 20, 255), 3)
    viewImage(output, "Two cats split by line")
    '''

    ''' adding text to image

    image = cv2.imread("./images/cat_text.jpg")
    output = image.copy()
    cv2.putText(output, "We <3 Cats", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (189, 0, 122), 5)
    viewImage(output, "Cat with text")
    '''

    #''' detect faces on pictures
    
    image_path = "images/faces.jpg"
    face_cascade = cv2.CascadeClassifier('C:/PythonProjects/opencv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    faces_detected = "Faces on picture detected: " + format(len(faces))
    # Рисуем квадраты вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (80, 160, 255), 2)
    viewImage(image, faces_detected)
    #'''
