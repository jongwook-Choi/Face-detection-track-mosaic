import cv2
from facenet_pytorch import MTCNN

'''
MTCNN을 활용한 실시간 Face detector 모듈
'''

class FaceDetector(object):
    """
    Face Detector class
    """
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw boundinf box, probs, landmarks
        """
        for box, prob, ld in zip(boxes, probs, landmarks):
            # draw rectangle
            cv2.rectangle(frame,            # 이미지
                          (box[0], box[1]), # 시작점 좌표 (x, y) -> 좌상단
                          (box[2], box[3]), # 종료점 좌표 (x, y) -> 우하단
                          (0, 0, 255),      # 빨간색
                          thickness=2)      # 선 두께

            # show probability
            # 이미지, 표시문자, 위치, 폰트, 글자크기, 색, 두께, 선 유형
            cv2.putText(frame, str(prob), (box[2], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # Draw landmarks -> 눈 코 입
            # 눈코입에 점 찍음
            cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 225), -1)
            cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 225), -1)
        return frame

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
        # 커널 : 이미지에서 (x, y)의 픽셀과 (x, y) 픽셀 주변을 포함한 작은 크기의 공간
        # -> cnn 의 커널과 유사 개념
        (h, w) = image.shape[:2]
        kW = int(w/factor)
        kH = int(h/factor)

        # Ensure width and height of kernel are odd
        # 커널은 홀수여야함
        if kW % 2 == 0:
            kW -= 1
        if kH % 2 == 0:
            kH -= 1

        # Apply a Gaussian blur to the input image using the computer kernel size
        # 가우시안 블러링 : 중심에 있는 픽셀에 높은 가중치를 부여
        return cv2.GaussianBlur(image, (kW, kH), 0)

    def run(self):
        # laptop webcam 을 사용할때 0
        cap = cv2.VideoCapture(0)

        while True:
            # 재생되는 비디오의 한 프레임씩 읽는다.
            # ret : boolean -> 프레임을 제대로 읽었을 때 : True
            # frame : frame -> 읽은 프레임
            ret, frame = cap.read()

            try:
                # boxes : 바운딩 박스
                # probs : 얼굴일 확률
                # landmarks : 좌 우 눈, 코, 입 좌 우
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self._draw(frame, boxes, probs, landmarks)
            except:
                pass

            # frame 을 화면에 디스프레이함.
            cv2.imshow('face detector', frame)

            # video 를 멈추고 싶을때 누를 키 설정 -> q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 오픈한 cap 객체를 cap.release() 함수를 이용해 해제
        cap.release()
        # 생성한 모든 윈도우를 제거
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mtcnn = MTCNN()
    fcd = FaceDetector(mtcnn)
    fcd.run()