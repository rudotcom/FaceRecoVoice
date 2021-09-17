import cv2
from prediction import predict
from show_labels import show_prediction_labels_on_image
from voice import say_hello


def capture(url, model_path):
    cap = cv2.VideoCapture(url)
    process_this_frame = 1
    predictions = None

    while True:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame += 1
            # process one frame in every 30 frames for speed
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path=model_path)

            if predictions:
                say_hello(predictions)
                frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
