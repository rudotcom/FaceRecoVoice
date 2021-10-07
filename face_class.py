import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import speechd

tts_d = speechd.SSIPClient('test')
tts_d.set_output_module('rhvoice')
tts_d.set_language('ru')
tts_d.set_rate(50)
tts_d.set_punctuation(speechd.PunctuationMode.SOME)
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# Use Truetype font to write Cyrillic.
fontpath = "fonts/20094.ttf"
font = ImageFont.truetype(fontpath, 45)
train_path = 'known_people/train'  # known people image folder
enc_path = 'known_people/encodings'  # known people encodings to be saved to
known_face_encodings = []


class Face:

    def __init__(self, encoding, name):
        self.encoding = encoding
        self.name = name
        self.last_seen = None

    def learn(self, img_path):
        self.name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'Обучение... {self.name}')

        image = face_recognition.load_image_file(img_path)
        print('--', os.path.basename(img_path))
        self.encoding = face_recognition.face_encodings(image)[0]
        faces.append(Face(self.encoding, self.name))

        with open(os.path.join(enc_path, self.name), "w") as f:
            for row in self.encoding:
                f.write(str(row))
                f.write('\n')


# Создать массив лиц
faces = []

for face_name in os.listdir(enc_path):
    encoding_file = os.path.join(enc_path, face_name)
    face_encoding = np.loadtxt(encoding_file)
    faces.append(Face(face_encoding, face_name))
    known_face_encodings.append(face_encoding)

# Инициализация переменных
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
names_seen = []
name = ''
# Получить ссылку на вебкамеру #0 (ту что по умолчанию)
video_capture = cv2.VideoCapture(0)

while True:
    # Захват кадра видео
    ret, frame = video_capture.read()
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    if ret:
        # Уменьшение размеров кадра для более быстрого распознавания
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Конвертация изображения из цветов BGR (которые испоользует OpenCV)
        # в цвета RGB (которые использует face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Обработка кадров через один для ускорения процесса
        if process_this_frame:
            # Найти все лица и их кодировки в текущем кадре видео
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face in faces:
                # Совпадает ли лицо с известными лицами
                matches = face_recognition.compare_faces(known_face_encodings, face.encoding)
                name = ""

                # Используем известное лицо с наименьшим расстоянием от нового лица
                face_distances = face_recognition.face_distance(known_face_encodings, face.encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    # name = face.from_encoding()
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Отобразить результаты
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            if name:
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (50, 220, 255), cv2.FILLED)
                # cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100, 10), 2)
                draw.text((left + 1, top - 65), name, font=font, fill=(0, 80, 255, 100))
                draw.text((left, top - 66), name, font=font, fill=(255, 255, 255, 10))
                frame = np.array(img_pil)
                if name not in names_seen:
                    tts_d.speak(f'Здравствуй, {name}')
                    names_seen.append(name)

            else:
                # Прямоугольник вокруг неизвестного лица лица
                cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100, 10), 2)
                draw.text((left + 7, bottom - 5), 'Кто это?', font=font, fill=(255, 255, 255, 100))

        # Отобразить получившийся кадр
        cv2.imshow('Video', frame)

        # Нажать 'q' на клавиатуре, чтобы выйти!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освободить связь с вебкамерой
tts_d.close()
video_capture.release()
cv2.destroyAllWindows()
