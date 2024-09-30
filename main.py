#from hand_detection import hand_detection
from gesture_recognition import recognize_gesture
from perform_actions import perform_action
import cv2

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hand and recognize gesture
        hand_frame = hand_detection(frame)
        gesture_class = recognize_gesture(hand_frame)

        # Perform action based on gesture
        perform_action(gesture_class)

        # Display the video feed
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
