import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
from model.inference import ModelInferencer

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandWatcher():
    def __init__(self) -> None:
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...


    def writeChord(self, chord, img):
        cv2.putText(img, chord,
                    placement = (10, 500),
                    font=cv2.FONT_HERSHERY_SIMPLEX,
                    fontScale=1, 
                    fontColor=(255, 255, 255),
                    thickness=1,
                    lineType=2)
        
    def cropHands(self, videoPath):
        model = ModelInferencer()
        handCordAmount = 21
        padding = 50
        xHandCords = [None]*handCordAmount
        yHandCords = [None]*handCordAmount
        # For webcam input:
        cap = cv2.VideoCapture(videoPath)
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                image_height, image_width, _ = image.shape
                #image = cv2.flip(image, 1)
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
                            cx, cy = landmrk.x, landmrk.y
                            xHandCords[ids] = cx
                            yHandCords[ids] = cy

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
                        cv2.imwrite(f"/tmp/currFrame.jpg", handCropped)
                        handImage = model.convertImage(f"/tmp/currFrame.jpg")
                        chordShape= model.classifyPrediction(model.predict(handImage))
                        #print(chordShape)
                        cv2.putText(image, chordShape,org=(10, 500),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3, color=(0, 255, 0),thickness=3,lineType=2)
                        # self.writeChord(chordShape, image)
                        # self.writeChord(chordShape, handCropped)
                        cv2.imshow('Cropped Hands', handCropped)
                    if yHandCordMax > image_height or xHandCordMax > image_width or xHandCordMin< padding or yHandCordMin < padding:
                        cv2.destroyWindow('Cropped Hands')
                    
                cv2.imshow('MediaPipe Hands', image)
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break  
        cap.release()
        
    def inference(self, handImage):
        pass

    def writeTextOnImage(self, text, img):
        cv2.putText(img, text,
                    placement = (10, 500),
                    font=cv2.FONT_HERSHERY_SIMPLEX,
                    fontScale=1, 
                    fontColor=(255, 255, 255),
                    thickness=1,
                    lineType=2)

