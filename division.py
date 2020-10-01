import cv2 as cv


def division(input_img):
    # preview img
    # cv.namedWindow("img", cv.WINDOW_KEEPRATIO)
    # cv.imshow("img", input_img)
    # cv.waitKey(0)

    # preprocess
    area_min_threshold = 100
    area_max_threshold = 2000
    gray_res = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(gray_res, (5, 5), 0)
    _, binary = cv.threshold(gray_res, 100, 255, cv.THRESH_BINARY_INV)
    # edged = cv.Canny(blurred, 30, 150)

    # get contours
    output_contours = []
    output_rects = []
    output_rois = []
    contours, hierarchy = cv.findContours(binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        print("Find nothing")
    for contour in contours:
        tmp_area = cv.contourArea(contour)
        if area_min_threshold <= tmp_area <= area_max_threshold:
            output_contours.append(contour)
            output_rects.append(cv.boundingRect(contour))
    for rect in output_rects:
        (x, y, w, h) = rect
        # cv.rectangle(input_img, (x, y), (x + w, y + h), (0, 125, 0), 15)
        ROI = binary[y:(y + h), x:(x + w)]
        (height, width) = ROI.shape[:2]
        if height > width:
            diff = height - width
            ROI = cv.copyMakeBorder(ROI, 0, 0, int(diff / 2), int(diff / 2), cv.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            diff = width - height
            ROI = cv.copyMakeBorder(ROI, int(diff / 2), int(diff / 2), 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
        ROI = cv.resize(ROI, (32, 32), interpolation=cv.INTER_NEAREST)
        output_rois.append(ROI)
    # cv.drawContours(input_img, output_contours, -1, (255, 0, 0), 20)
    # cv.namedWindow("res", cv.WINDOW_NORMAL)
    # cv.imshow("res", input_img)
    # cv.waitKey(0)

    # cv.destroyAllWindows()
    return output_rois, output_rects
