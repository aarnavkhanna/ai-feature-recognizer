import cv2
import mediapipe as mp


drawing_utils = mp.solutions.drawing_utils
holistic_model = mp.solutions.holistic

camera = cv2.VideoCapture(0)

with holistic_model.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as model:
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection_results = model.process(rgb_image)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        drawing_utils.draw_landmarks(
            bgr_image, detection_results.pose_landmarks, holistic_model.POSE_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(100, 50, 20), thickness=3, circle_radius=5),
            drawing_utils.DrawingSpec(color=(100, 80, 140), thickness=3, circle_radius=3)
        )

        drawing_utils.draw_landmarks(
            bgr_image, detection_results.left_hand_landmarks, holistic_model.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(140, 30, 90), thickness=3, circle_radius=5),
            drawing_utils.DrawingSpec(color=(140, 60, 255), thickness=3, circle_radius=3)
        )

        drawing_utils.draw_landmarks(
            bgr_image, detection_results.right_hand_landmarks, holistic_model.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(255, 130, 70), thickness=3, circle_radius=5),
            drawing_utils.DrawingSpec(color=(255, 70, 240), thickness=3, circle_radius=3)
        )

        drawing_utils.draw_landmarks(
            bgr_image, detection_results.face_landmarks, holistic_model.FACEMESH_TESSELATION,
            drawing_utils.DrawingSpec(color=(100, 120, 20), thickness=2, circle_radius=2),
            drawing_utils.DrawingSpec(color=(100, 260, 130), thickness=2, circle_radius=2)
        )

        cv2.imshow('Webcam View', bgr_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
