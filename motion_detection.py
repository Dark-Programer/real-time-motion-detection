import cv2 as cv
path = "Computer Vision/motion-filtering/walking_people.mp4"
video = cv.VideoCapture(path)
web_cam = cv.VideoCapture(0)

frequency = 120
threshold = 16
subtractor = cv.createBackgroundSubtractorMOG2(frequency, threshold)

while True:
    return_value, frame = video.read()
    webcam_return_value, webcam_frame = web_cam.read()
    if return_value or webcam_return_value:
        mask = subtractor.apply(cv.resize(frame, (640, 480)))
        web_mask = subtractor.apply(cv.resize(webcam_frame, (640, 480)))

        cv.imshow("mask", mask)
        cv.imshow("web_mask", web_mask)

        # When user want to break the loop he will press x
        if cv.waitKey(5) == ord("x"):
            break

    else:
        video = cv.VideoCapture(path)
        web_cam = cv.VideoCapture(0)


cv.destroyAllWindows()
video.release()
