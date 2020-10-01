from division import division
import tensorflow as tf
import numpy as np
import cv2 as cv

model = tf.keras.models.load_model("model")
cap = cv.VideoCapture(0)
cv.namedWindow("result", cv.WINDOW_KEEPRATIO)

if __name__ == '__main__':
    # img = cv.imread("digits.jpg")
    # _, frame = cap.read()

    while True:
        _, frame = cap.read()
        rois, rects = division(frame)
        if len(rois) == 0:  # if there is no roi.
            cv.imshow("result", frame)
            key = cv.waitKey(23)
            if key == 27:
                break
            continue

        data = np.array(rois)
        data = data / 255.0
        # print(data)

        results = model.predict(data)
        predicted_labels = []
        for result in results:
            predicted_labels.append(np.argmax(result))

        if len(rects) == len(predicted_labels):
            for i in range(len(rects)):
                x, y, w, h = rects[i]
                cv.putText(frame, str(predicted_labels[i]), (x, y), cv.FONT_HERSHEY_PLAIN, 5,
                           (0, 100, 0), 5)
                cv.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 0), 2)

        print(predicted_labels)

        cv.imshow("result", frame)
        cv.resizeWindow("result", 1000, 700)
        key = cv.waitKey(23)
        if key == 27:
            break

    cv.destroyAllWindows()
    cap.release()
