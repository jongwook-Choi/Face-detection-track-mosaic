import splitfolders

'''
이미지 데이터를 train 과 val 로 나눠줌
ratio 를 추가하면 test 도 추가 가능
'''
input_folder_path = '../data/frames'
output_folder_path = '../data'

if __name__ == '__main__':
    splitfolders.ratio(input_folder_path, output=output_folder_path, seed=1337, ratio=(0.9, 0.1))