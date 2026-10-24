
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# JARVIS Hand Gesture Cursor
# Press Q in the camera window to quit.

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

screen_w, screen_h = pyautogui.size()
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam could not be opened.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Adjust these if tracking is unstable.
SMOOTHING = 0.25
PINCH_THRESHOLD = 0.045
FRAME_MARGIN = 0.08

previous_x = screen_w / 2
previous_y = screen_h / 2
pinch_was_active = False
last_click_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            # Mirror the camera preview like a mirror.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            status = "Show your hand"

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                index_tip = hand.landmark[8]
                thumb_tip = hand.landmark[4]

                # Keep the pointer away from extreme camera edges.
                nx = (index_tip.x - FRAME_MARGIN) / (
                    1 - 2 * FRAME_MARGIN
                )
                ny = (index_tip.y - FRAME_MARGIN) / (
                    1 - 2 * FRAME_MARGIN
                )
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))

                target_x = nx * (screen_w - 1)
                target_y = ny * (screen_h - 1)

                # Smooth motion to reduce jitter.
                current_x = (
                    previous_x + SMOOTHING * (target_x - previous_x)
                )
                current_y = (
                    previous_y + SMOOTHING * (target_y - previous_y)
                )

                previous_x = current_x
                previous_y = current_y

                pyautogui.moveTo(int(current_x), int(current_y))

                distance = math.hypot(
                    index_tip.x - thumb_tip.x,
                    index_tip.y - thumb_tip.y
                )
                pinch_active = distance < PINCH_THRESHOLD

                # Click only when a new pinch begins.
                now = time.time()
                if (
                    pinch_active
                    and not pinch_was_active
                    and now - last_click_time > 0.45
                ):
                    pyautogui.click()
                    last_click_time = now

                pinch_was_active = pinch_active
                status = "PINCH: click" if pinch_active else "Tracking"

                px = int(index_tip.x * width)
                py = int(index_tip.y * height)
                cv2.circle(frame, (px, py), 10, (255, 100, 0), -1)

            else:
                # Require releasing the pinch before the next click.
                pinch_was_active = False

            cv2.putText(
                frame, "JARVIS - Hand Gesture Cursor",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2
            )
            cv2.putText(
                frame, "Press Q to quit",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.imshow("JARVIS Hand Gesture Cursor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
