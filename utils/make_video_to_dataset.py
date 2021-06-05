import cv2
from facenet_pytorch import MTCNN
import os
import argparse
'''
내 얼굴 식별을 위해 동영상을 frame 별로 짤라주는 모듈
'''
# argparse 로 하던가 아님 config 파일로 정리
data_root = "../data"
name = 'wook'
count = 250

class MakeV2D:
    def __init__(self, name=name, data_root=data_root, count=count):
        self.data_root = data_root
        if name == "wook":
            self.name = "wook"
            self.count = False
        else:
            self.name = "not_wook"
            self.count = count
        self.video_root = os.path.join(data_root, "videos")
        self.mtcnn = MTCNN()

    def run(self):
        # Load the video
        video_dir = os.path.join(self.video_root, self.name)
        video_list = os.listdir(video_dir)

        if os.path.isfile(video_dir):
            raise Exception('root is empty!')

        frame_num_f = 0
        for video in video_list:
            print(f'{video} in process')
            if video[-4:-1] != '.mp4':
                pass
            video_path = os.path.join(video_dir, video)

            # Load the video
            v_cap = cv2.VideoCapture(video_path)
            # get the frame count
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(v_len)

            frames = []

            for _ in range(v_len):
                success, frame = v_cap.read()  # Load the frame

                if not success:
                    continue
                # cv2.COLOR_BGR2RGB: https://crmn.tistory.com/49 참고
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frames_dir = os.path.join(self.data_root, "frames")
            frames_name = os.path.join(frames_dir, self.name)
            if not os.path.exists(frames_name):
                os.makedirs(frames_name)

            frame_num_b = frame_num_f
            if self.count:
                frame_num_f += self.count
                if self.count > len(frames):
                    assert Exception('Too many count!')
            else:
                frame_num_f += len(frames)
            frames_path = [f'{frames_name}/{self.name}_{i}.jpg' for i in range(frame_num_b+1, frame_num_f+1)]

            for frame_, path_ in zip(frames, frames_path):
                self.mtcnn(frame_, save_path=path_)


def main():
    Video2Dataset = MakeV2D('notwook')
    Video2Dataset.run()


if __name__ == '__main__':
    main()