import cv2
import mediapipe as mp
import sys
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

handCordAmount = 21
padding = 100
xHandCords = [None]*handCordAmount
yHandCords = [None]*handCordAmount
count = 0
videoName = "minor"
videoPath = f"./data_capture/videos/{videoName}.mov"
framesPath = f"./data_capture/captured_frames/{videoName}"

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    print(idx)
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(videoPath)
if not os.path.exists(framesPath):
        os.mkdir(framesPath)
#cap = cv2.VideoCapture(0) # Use this to utilize webcam capturing
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image_height, image_width, _ = image.shape
    #image = cv2.flip(image, 0)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            # print(ids, landmrk)
            #cx, cy = landmrk.x * image_width, landmrk.y*image_height
            cx, cy = landmrk.x, landmrk.y
            xHandCords[ids] = cx
            yHandCords[ids] = cy
            #print(cx, cy)
            # print (ids, cx, cy)
      xHandCordMax = int(max(xHandCords)*image_width)
      yHandCordMax = int(max(yHandCords)*image_height)
      xHandCordMin = int(min(xHandCords)*image_width)
      yHandCordMin = int(min(yHandCords)*image_height)
      cropX = xHandCordMin-padding
      cropWidth = xHandCordMin + (xHandCordMax-xHandCordMin) + padding
      cropY = yHandCordMin-padding
      cropHeight = yHandCordMin + (yHandCordMax-yHandCordMin) + padding
      if (xHandCordMin > padding and yHandCordMin > padding):
        handCropped = image[cropY:cropHeight, cropX:cropWidth]
        cv2.imshow('Cropped Hands', handCropped)

        cv2.imwrite(f"{framesPath}/{videoName}_{count}.jpg", handCropped)
        print(f'Captured frame {count}: ', success)
        count+=1
      if yHandCordMax > image_height or xHandCordMax > image_width or xHandCordMin< padding or yHandCordMin < padding:
        cv2.destroyWindow('Cropped Hands')
    cv2.imshow('MediaPipe Hands', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break  
cap.release()