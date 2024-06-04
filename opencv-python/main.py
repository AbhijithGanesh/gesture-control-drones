import copy
from collections import deque
import cv2 as cv
import mediapipe as mp
import threading
from args import get_args
from classifier import HandsClassifier
from controller import MavHandler
from cvUtils import (
    calc_bounding_rect,
    calc_landmark_list,
    draw_bounding_rect,
    draw_info_text,
    draw_landmarks,
    draw_point_history,
    pre_process_landmark,
    select_mode,
)


def main():
    kc_labels = ["Open", "Close", "Pointer", "OK"]
    mav = MavHandler()
    mav.handle_mode("GUIDED")
    mav.handle_arming()
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_img_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_img_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    kypoint_classifier = HandsClassifier()
    ptHistory = deque(maxlen=16)
    frame_skip = 2
    frame_count = 0

    takeoff_thread = None
    land_thread = None
    stop_threads = threading.Event()

    def handle_takeoff():
        mav.handle_takeoff((37.6200, -122.3750, 5))

    def handle_land():
        mav.handle_mode("RTL")

    while True:
        ky = cv.waitKey(10)
        if ky == 27:  # ESC
            break

        ret, img = cap.read()
        if not ret:
            break

        if frame_count % (frame_skip + 1) == 0:
            img = cv.flip(img, 1)  # Mirror display
            dbgImg = img.copy()  # Use a shallow copy instead of deepcopy
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    brect = calc_bounding_rect(dbgImg, hand_landmarks)
                    landmark_list = calc_landmark_list(dbgImg, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    hand_sign_id = kypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 2:  # Point gesture
                        ptHistory.append(landmark_list[8])
                    else:
                        ptHistory.append([0, 0])

                    dbgImg = draw_bounding_rect(use_brect, dbgImg, brect)
                    dbgImg = draw_landmarks(dbgImg, landmark_list)
                    if kc_labels[hand_sign_id] == "Open":
                        if land_thread and land_thread.is_alive():
                            stop_threads.set()
                            land_thread.join()
                            stop_threads.clear()

                        if takeoff_thread is None or not takeoff_thread.is_alive():
                            takeoff_thread = threading.Thread(target=handle_takeoff)
                            takeoff_thread.start()
                        dbgImg = draw_info_text(
                            dbgImg, brect, handedness, kc_labels[hand_sign_id]
                        )
                    elif kc_labels[hand_sign_id] == "Pointer":
                        if takeoff_thread and takeoff_thread.is_alive():
                            stop_threads.set()
                            takeoff_thread.join()
                            stop_threads.clear()

                        if land_thread is None or not land_thread.is_alive():
                            land_thread = threading.Thread(target=handle_land)
                            land_thread.start()
                        dbgImg = draw_info_text(
                            dbgImg, brect, handedness, kc_labels[hand_sign_id]
                        )
            else:
                ptHistory.append([0, 0])

            dbgImg = draw_point_history(dbgImg, ptHistory)
            cv.imshow("Hand Gesture Recognition", dbgImg)

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
