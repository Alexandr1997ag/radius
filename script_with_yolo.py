import numpy as np
import torch
import cv2 as cv
import mediapipe as mp
import torch
from mediapipe.framework.formats import landmark_pb2

class System:
    def __init__(self):
        device = torch.device('cuda')
        self.path_to_yolo_dir = input("Введите абсолютный путь до директории yolov5 ")
        self.path_to_weights = input("Введите абсолютный путь до весов обученной нейросети (до файла weights.pt) ")
        self.path_to_video = input("Введите абсолютный путь до видео, которое хотите запустить ")
        self.model = torch.hub.load(self.path_to_yolo_dir, 'custom', source='local', path=self.path_to_weights,
                               force_reload=True)

    def params(self, cap):
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(5)
        size = (frame_width, frame_height)
        return frame_width, frame_height, fps, size


    def ball_det(self):
        cap = cv.VideoCapture(self.path_to_video)
        frame_width, frame_height, fps, size = self.params(cap)
        result = cv.VideoWriter('filename_1_2.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                fps, size)
        while (1):
            ret, frame = cap.read()

            if ret == False:
                cap.release()
                cv.destroyAllWindows()
                break
            try:
                results = self.model(frame)
                result.write(np.squeeze(results.render()))
                print("происходит обработка видео детекции мяча... ")
                # cv.imshow('VIDEO', np.squeeze(results.render()))

                # cv.imshow(f"frame", results)
            except Exception as e:
                print(str(e))
                continue
            if cv.waitKey(10) == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                break

    def pose(self):
        cap = cv.VideoCapture('filename_1_2.avi')
        frame_width, frame_height, fps, size = self.params(cap)
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        result = cv.VideoWriter('./filename_pose_1.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                fps, size)

        while (1):
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                cv.destroyAllWindows()
                break
            try:
                # # draw landmarks (если хотим распознавать все точки и отрисовывать между ними линии)
                # mp_drawing.draw_landmarks(
                #     frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                results = pose.process(frame)
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        # results.pose_landmarks.landmark[0],
                        results.pose_landmarks.landmark[11],
                        results.pose_landmarks.landmark[12],
                        results.pose_landmarks.landmark[13],
                        results.pose_landmarks.landmark[14],
                        results.pose_landmarks.landmark[15],
                        results.pose_landmarks.landmark[16],
                        results.pose_landmarks.landmark[17],
                        results.pose_landmarks.landmark[18],
                        results.pose_landmarks.landmark[19],
                        results.pose_landmarks.landmark[20],
                        results.pose_landmarks.landmark[21],
                        results.pose_landmarks.landmark[22],
                        results.pose_landmarks.landmark[23],
                        results.pose_landmarks.landmark[24],
                        results.pose_landmarks.landmark[25],
                        results.pose_landmarks.landmark[26],
                        results.pose_landmarks.landmark[27],
                        results.pose_landmarks.landmark[28],
                        results.pose_landmarks.landmark[29],
                        results.pose_landmarks.landmark[30],
                        results.pose_landmarks.landmark[31],
                        results.pose_landmarks.landmark[32]
                    ]
                )

                mp_drawing.draw_landmarks(
                    # frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)
                    frame, landmark_list=landmark_subset)

                result.write((frame))
                print("Происходит обработка позы человека...")

                # cv.imshow(f"frame", frame)
            except Exception as e:
                print(str(e))
                continue
            if cv.waitKey(10) == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                break

a = System()

if __name__ == "__main__":
    a.ball_det()
    a.pose()


# /media/alex/One Touch1/Radius/yolov5
# /media/alex/One Touch1/Radius/weights.pt
# /media/alex/One Touch1/Radius/150 видео-20230214T184524Z-001/150 видео/oYDtBfO8A6yINzPaUFJAZ74tuChC2xzowQAwIk.mp4