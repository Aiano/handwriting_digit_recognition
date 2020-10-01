import cv2 as cv
from division import division

res_binaries = []
num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

if __name__ == '__main__':
    # cap = cv.VideoCapture(0)
    #
    # img = None
    # while True:
    #     _, img = cap.read()
    #     cv.imshow("img", img)
    #
    #     key = cv.waitKey(23)
    #     if key == ord('p'):
    #         break
    # cv.destroyAllWindows()
    img = cv.imread("source_imgs/9.jpg")

    rois, rects = division(img)
    print(rois)
    for roi in rois:
        cv.imshow("roi", roi)
        cv.waitKey(30)

        label = input("label:")
        if int(label) != -1:
            num[int(label)] += 1
            address = "data/" + label + "/" + str(num[int(label)]) + ".png"
            cv.imwrite(address, roi)
    cv.destroyAllWindows()
