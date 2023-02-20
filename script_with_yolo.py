# YOLOv5 PyTorch HUB Inference (DetectionModels only)
import numpy as np
import torch
import cv2 as cv
from vidgear.gears import WriteGear

device = torch.device('cuda')
path_to_yolo_dir = input("Введите абсолютный путь до директории yolov5 ")
path_to_weights = input("Введите абсолютный путь до весов обученной нейросети (до файла weights.pt) ")
path_to_video = input("Введите абсолютный путь до видео, которое хотите запустить ")
model = torch.hub.load(path_to_yolo_dir, 'custom', source='local', path=path_to_weights, force_reload=True)
# model = torch.hub.load("/media/alex/One Touch1/Radius/yolov5", 'custom', source='local', path='/media/alex/One Touch1/Radius/weights.pt', force_reload=True)


# cap = cv.VideoCapture('/media/alex/One Touch1/Radius/150 видео-20230214T184524Z-001/150 видео/oYDtBfO8A6yINzPaUFJAZ74tuChC2xzowQAwIk.mp4')
cap = cv.VideoCapture(path_to_video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
size = (frame_width, frame_height)
result = cv.VideoWriter('filename.avi',
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps, size)




while(1):
    ret, frame = cap.read()

    if ret == False:
        cap.release()
        cv.destroyAllWindows()
        break
    try:
        results = model(frame)
        result.write(np.squeeze(results.render()))
        print("происходит обработка видео... ")
        # cv.imshow('VIDEO', np.squeeze(results.render()))

        # cv.imshow(f"frame", results)
    except Exception as e:
        print(str(e))
        continue
    if cv.waitKey(10) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break


# /media/alex/One Touch1/Radius/yolov5
# /media/alex/One Touch1/Radius/weights.pt
# /media/alex/One Touch1/Radius/150 видео-20230214T184524Z-001/150 видео/oYDtBfO8A6yINzPaUFJAZ74tuChC2xzowQAwIk.mp4

