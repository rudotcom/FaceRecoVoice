import os

from train_model import train
from capture import capture

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    ip_cam = '192.168.1.209'
    ip_cam_url = f'http://{ip_cam}:81/stream'
    known_people = 'known_people'
    model_path = f"{known_people}/trained_knn_model.clf"

    if os.path.exists(model_path):
        print('Модели уже обучены. Использую их...')
    else:
        print("Обучаю классификаторы KNN...")
        classifier = train(f"{known_people}/train", model_save_path=model_path, n_neighbors=2)
        print("Обучение завершено!")

    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Ожидание камеры...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'

    capture(ip_cam_url, model_path)
