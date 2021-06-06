import os
import cv2, time
from facenet_pytorch import MTCNN
import argparse

data_dir = '../data'
video_root = os.path.join(data_dir, 'videos/wook')
file_name = 'wook_video.mp4'
save_dir = os.path.join(data_dir, 'result/demo_bbox')


class DemoMTCNN(object):
    def __init__(self, detector, args):
        self.detector = detector
        self.args = args
        self.video_path = self.args.video_path
        self.save_path = self.args.save_path

    def _draw(self, frame, boxes, probs, landmarks):
        for box, prob, ld in zip(boxes, probs, landmarks):
            if prob < 0.5: # MOT video 에 대해 높임
                continue
            # draw rectangle
            # 에러 발생할 경우
            # opencv-python 4.5.1.48 version 설치
            cv2.rectangle(frame, # 이미지
                          (box[0], box[1]),  # 시작점 좌표 (x, y) -> 좌상단
                          (box[2], box[3]),  # 종료점 좌표 (x, y) -> 우하단
                          (0, 0, 255),  # 빨간색
                          thickness=2)  # 선 두께
            # show probability
            # cv2.putText(이미지, 표시문자, 위치, 폰트, 글자크기, 색, 두께, 선 유형)
            cv2.putText(frame, f"{prob:.4f}", (box[2], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            # Draw landmarks -> 눈 코 입
            # 눈코입에 점 찍음
            cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 225), -1)

        return frame

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
                result_img = self._draw(result_img, boxes, probs, landmarks)

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
    fcd = DemoMTCNN(detector, args)
    fcd.run()


