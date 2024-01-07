import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)  # Fetching video camera.
# Setting video capture resolution for the webcam to 1280x720 (px)
cap.set(3, 1280)
cap.set(4, 720)

mp_drawing = mp.solutions.drawing_utils

# Fetching hand models.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def write_text(img, text, x, y):
    """ Writing (overlaying) text on the OpenCV camera view.

    Parameters
    __________
    img:
        Frame of the current image.
    text: str
        Text being written to the camera view.
    x: int
        X coordinate for plotting the text.
    y: int
        Y coordinate for plotting the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (x, y)
    fontScale = 1
    fontColor = (255, 255, 255)  # White.
    lineType = 2
    cv2.putText(img, text, pos, font, fontScale, fontColor, lineType)

# Used for FPS calculations.
def steering_wheel():
    prev_frame_time = 0
    new_frame_time = 0
    while cap is not None:
        success, img = cap.read()
        if not success:
            break
        cv2.waitKey(1)  # Continuously refreshes the webcam frame every 1ms.
        img = cv2.flip(img, 1)
        img.flags.writeable = False  # Making the images not writeable for optimization.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Processing video.

        # Checking if a hand exists in the frame.
        landmarks = results.multi_hand_landmarks  # Fetches all the landmarks (points) on the hand.
        if landmarks:
            if len(landmarks) == 2:  # If 2 hands are in view.
                left_hand_landmarks = landmarks[1].landmark
                right_hand_landmarks = landmarks[0].landmark

            # Define the landmarks for the left hand
                left_thumb_tip = left_hand_landmarks[4]
                left_index_tip = left_hand_landmarks[8]
                left_middle_tip = left_hand_landmarks[12]
                left_ring_tip = left_hand_landmarks[16]
                left_pinky_tip = left_hand_landmarks[20]

                # Define the landmarks for the right hand
                right_thumb_tip = right_hand_landmarks[4]
                right_index_tip = right_hand_landmarks[8]
                right_middle_tip = right_hand_landmarks[12]
                right_ring_tip = right_hand_landmarks[16]
                right_pinky_tip = right_hand_landmarks[20]

                # Get the y-coordinates of the thumb tips for both hands
                left_thumb_tip_y = left_thumb_tip.y
                right_thumb_tip_y = right_thumb_tip.y

                # Check if both hands are closed
                is_left_hand_closed = (
                    left_index_tip.y > left_thumb_tip_y and
                    left_middle_tip.y > left_thumb_tip_y and
                    left_ring_tip.y > left_thumb_tip_y and
                    left_pinky_tip.y > left_thumb_tip_y
                    )

                is_right_hand_closed = (
                    right_index_tip.y > right_thumb_tip_y and
                    right_middle_tip.y > right_thumb_tip_y and
                    right_ring_tip.y > right_thumb_tip_y and
                    right_pinky_tip.y > right_thumb_tip_y
                    )

# Now you have boolean values 'is_left_hand_closed' and 'is_right_hand_closed' to check if both hands are closed.


                if is_left_hand_closed or is_right_hand_closed :
                    print("Accelerate")
                    sensitivity = 0.1  # Adjusts sensitivity for turning; the higher this is, the more you have to turn your hands.
                    # Calculate slope between middle fingers of both hands
                    left_mFingerX, left_mFingerY = left_hand_landmarks[11].x, left_hand_landmarks[11].y
                    right_mFingerX, right_mFingerY = right_hand_landmarks[11].x, right_hand_landmarks[11].y
                    slope = (right_mFingerY - left_mFingerY) / (right_mFingerX - left_mFingerX)

                    if abs(slope) > sensitivity:
                        if slope < 0:
                            # When the slope is negative, we turn left.
                            print("Turn left.")
                            write_text(img, "Left.", 360, 360)
                            pyautogui.keyDown("left")
                            pyautogui.keyUp("right")
                        if slope > 0:
                            # When the slope is positive, we turn right.
                            print("Turn right.")
                            write_text(img, "Right.", 360, 360)
                            pyautogui.keyDown("right")
                            pyautogui.keyUp("left")
                    if abs(slope) < sensitivity:
                        # When our hands are straight, we stay still (and also throttle).
                        print("Keeping straight.")
                        write_text(img, "Straight.", 360, 360)
                        pyautogui.keyUp("s")
                        pyautogui.keyUp("right")
                        pyautogui.keyUp("left")
                        pyautogui.keyDown('w')  # Remove this if you have pedals to control the speed

                    # Check if both thumbs are raised and press the spacebar
                    #if left_thumb_tip.y < left_index_tip.y or right_thumb_tip.y < right_index_tip.y:
                    #    print("Nitro Boost.")
                    #    pyautogui.press('space')
                else:
                    print("Reverse")
                    # Press 'S' when the hand is closed.
                    pyautogui.keyUp("w")
                    pyautogui.keyDown('s')

            # Iterating through landmarks (i.e., coordinates for finger joints) and drawing the connections.
            for hand_landmarks in landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # FPS Calculations
        new_frame_time = time.time()
        fps = str(int(1 / (new_frame_time - prev_frame_time)))
        write_text(img, fps, 150, 500)
        prev_frame_time = new_frame_time

        cv2.imshow("Hand Recognition", img)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the steering_wheel function to start the application
steering_wheel()
