import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
from model.inference import ModelInferencer
from data_capture.object_detector.neckDetect import LineDetector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class HandWatcher():
    def __init__(self) -> None:
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...
        self.loggedChords = []
        self.handCordAmount = 21
        self.padding = 100
        self.xHandCords = [None]*self.handCordAmount
        self.yHandCords = [None]*self.handCordAmount
        self.currFrame = '/tmp/currFrame.jpg'  # temporary frame to store captured hand image for referencing to ML model
    
    def images(self, IMAGE_FILES):
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
                cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
                # Draw hand world landmarks.
                if not results.multi_hand_world_landmarks:
                    continue
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    def cropHand(self, image):
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            image_height, image_width, _ = image.shape

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
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        cx, cy = landmrk.x, landmrk.y
                        self.xHandCords[ids] = cx
                        self.yHandCords[ids] = cy
                    xHandCordMax = int(max(self.xHandCords)*image_width)
                    yHandCordMax = int(max(self.yHandCords)*image_height)
                    xHandCordMin = int(min(self.xHandCords)*image_width)
                    yHandCordMin = int(min(self.yHandCords)*image_height)
                    cropX = xHandCordMin-self.padding
                    cropWidth = xHandCordMin + (xHandCordMax-xHandCordMin) + self.padding
                    cropY = yHandCordMin-self.padding
                    cropHeight = yHandCordMin + (yHandCordMax-yHandCordMin) + self.padding
                    if (xHandCordMin > self.padding and yHandCordMin > self.padding):
                        handCropped = image[cropY:cropHeight, cropX:cropWidth]
                        return handCropped
                return image

    def getChordShapes(self):
        return self.loggedChords
    
    def detectChordShapes(self, videoPath):
        model = ModelInferencer()
        xHandCords = [None]*self.handCordAmount
        yHandCords = [None]*self.handCordAmount
        # For webcam input:
        cap = cv2.VideoCapture(videoPath)
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    #self.detectChordShapes(videoPath)
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                image_height, image_width, _ = image.shape
                # if videoPath == 1:
                #     imageFlip = 1 # 0 flip around x axis, 1 flip around y axis, -1 flip around both axiz
                #     image = cv2.flip(image, imageFlip)
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
                        #### Commenting out below code to hide hand landmarks
                        # mp_drawing.draw_landmarks(
                        #     image,
                        #     hand_landmarks,
                        #     mp_hands.HAND_CONNECTIONS,
                        #     mp_drawing_styles.get_default_hand_landmarks_style(),
                        #     mp_drawing_styles.get_default_hand_connections_style())
                        for ids, landmrk in enumerate(hand_landmarks.landmark):
                            cx, cy = landmrk.x, landmrk.y
                            xHandCords[ids] = cx
                            yHandCords[ids] = cy

                    xHandCordMax = int(max(xHandCords)*image_width)
                    yHandCordMax = int(max(yHandCords)*image_height)
                    xHandCordMin = int(min(xHandCords)*image_width)
                    yHandCordMin = int(min(yHandCords)*image_height)
                    cropX = xHandCordMin-self.padding
                    cropWidth = xHandCordMin + (xHandCordMax-xHandCordMin) + self.padding
                    cropY = yHandCordMin-self.padding
                    cropHeight = yHandCordMin + (yHandCordMax-yHandCordMin) + self.padding
                    if (xHandCordMin > self.padding and yHandCordMin > self.padding):
                        #get timestamp
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                        #crop hand from image and reference it to model to get predicted chord shape
                        #handCropped = LineDetector().run(image[cropY:cropHeight, cropX:cropWidth])
                        handCropped = image[cropY:cropHeight, cropX:cropWidth]
                        extraCropped = LineDetector().run(handCropped)
                        cv2.imshow('Cropped Hands', handCropped)
                        cv2.imshow('Extra Crop', extraCropped)
                        #handCropped = LineDetector().run(handCropped)
                        cv2.imwrite(self.currFrame, handCropped)
                        handImage = model.convertImage(self.currFrame)
                        chordShape= model.classifyPrediction(model.predict(handImage))
                        #append timestamp and chordShape to list
                        self.loggedChords.append([chordShape, timestamp])

                        # Write predicted chord shape to frame and display frame
                        cv2.putText(image, chordShape,org=(10, 500),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3, color=(0, 255, 0),thickness=3,lineType=2)
                        # cv2.imshow('Cropped Hands', handCropped)

                        #Draw hand landmarks
                        mp_drawing.draw_landmarks(image, 
                                                  hand_landmarks,
                                                  mp_hands.HAND_CONNECTIONS, 
                                                  mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                  mp_drawing_styles.get_default_hand_connections_style())

                    # Check if hands are outside of bounds to prevent program from crashing
                    if yHandCordMax > image_height or xHandCordMax > image_width or xHandCordMin< self.padding or yHandCordMin < self.padding:
                        cv2.destroyWindow('Cropped Hands')
                    
                cv2.imshow('MediaPipe Hands', image)
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break  
        cap.release()


    

