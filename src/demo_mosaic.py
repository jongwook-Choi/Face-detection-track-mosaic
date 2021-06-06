import os
import cv2, time
from facenet_pytorch import MTCNN
import argparse
from demo_MTCNN import DemoMTCNN

data_dir = '../data'
video_root = os.path.join(data_dir, 'videos/wook')
file_name = 'wook_video.mp4'
save_dir = os.path.join(data_dir, 'result/demo_blur')

# DemoMTCNN 클래스 상속
# run 메서드는 오버라이팅
class DemoMosaic(DemoMTCNN):
    # 기반클래스 초기화
    def __init__(self, detector, args):
        self.detector = detector
        self.args = args
        self.video_path = self.args.video_path
        self.save_path = self.args.save_path

    def _detect_ROIs(self, boxes):
        """
        Return ROIs as a list
        (x1, x2, y1, y2)
        """
        ROIs = []
        for box in boxes:
            ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIs.append(ROI)

        return ROIs

    def _blur_face(self, image, factor=3.0):
        """
        Return blurred face
        """
        # Determine size of blurring kernel based on input image
        (h, w) = image.shape[:2]
        kW = int(w/factor)
        kH = int(h/factor)

        # Ensure width and height of kernel are odd

        if kW % 2 == 0:
            kW -= 1
        if kH % 2 == 0:
            kH -= 1

        # Apply a Gaussian blur to the input image using the computer kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        # fourcc 를 생성하여 디지털 미디어 포맷 코드를 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # cv2.VideoWriter("경로 및 제목", 비디오 포멧 코드, FPS, (녹화 파일 너비, 녹화파일 높이))
        out = cv2.VideoWriter(self.save_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frame_count, tt = 0, 0

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # prepare input
            result_img = img.copy()

            try:
                # inference, find face
                boxes, probs, landmarks = detector.detect(result_img, landmarks=True)
                # result_img = self._draw(result_img, boxes, probs, landmarks)

                ROIs = self._detect_ROIs(boxes)
                for roi, prob in zip(ROIs, probs):
                    if prob < 0.5:
                        continue
                    (startY, endY, startX, endX) = roi
                    face = result_img[startY:endY, startX:endX]

                    # run the blur function on the ROI
                    face = self._blur_face(face)
                    # blur recognized face area
                    result_img[startY:endY, startX:endX] = face

                # inference time
                tt += time.time() - start_time
                fps = frame_count / tt
                cv2.putText(result_img, f'FPS(mtcnn): {fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
            # visualize
            cv2.imshow('result', result_img)
            out.write(result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def parse_args(demo_video_path, demo_save_path):
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default=demo_video_path)
    parser.add_argument("--save_path", type=str, default=demo_save_path)

    return parser.parse_args()

if __name__ == '__main__':
    video_path = os.path.join(video_root, file_name)
    name = file_name.split("_")[0]
    save_name = f'{name}_demo.mp4'
    save_path = os.path.join(save_dir, save_name)
    args = parse_args(video_path, save_path)

    detector = MTCNN()
    fcd = DemoMosaic(detector, args)
    fcd.run()