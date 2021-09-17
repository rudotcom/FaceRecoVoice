import os

from train_model import train
from capture import capture

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    known_people = 'known_people'
    model_path = f"{known_people}/trained_knn_model.clf"

    if os.path.exists(model_path):
        print('Модели уже обучены. Использую их...')
    else:
        print("Обучаю классификаторы KNN...")
        classifier = train(f"{known_people}/train", model_save_path=model_path, n_neighbors=2)
        print("Обучение завершено!")

    print('Ожидание камеры...')
    capture(0, model_path)
