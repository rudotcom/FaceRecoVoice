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

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

train_path = 'known_people/train'  # known people image folder
enc_path = 'known_people/encodings'  # known people encodings to be saved to

# Loop through each person in the training set
for class_dir in os.listdir(train_path):
    if not os.path.isdir(os.path.join(train_path, class_dir)):
        continue

    known_name = os.path.splitext(os.path.basename(class_dir))[0]
    encoding_file = os.path.join(enc_path, known_name)
    known_face_names.append(known_name)

    if os.path.isfile(encoding_file):
        face_encoding = np.loadtxt(encoding_file)
        known_face_encodings.append(face_encoding)
        continue

    print(f'Обучение... {known_name}')
    # Loop through each training image for the current person
    for img_path in image_files_in_folder(os.path.join(train_path, class_dir)):
        image = face_recognition.load_image_file(img_path)
        print('--', os.path.basename(img_path))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        face_encoding_list = face_encoding.tolist
        with open(os.path.join(enc_path, known_name), "w") as f:
            for row in face_encoding:
                f.write(str(row))
                f.write('\n')

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
names_spoken = []
name = ''
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    if ret:
        # Resize frame of video to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            if name == '':
                cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100, 10), 2)
                draw.text((left + 7, bottom - 5), 'Кто это?', font=font, fill=(255, 255, 255, 100))

            elif name not in names_spoken:
                tts_d.speak(f'Здравствуй, {name}')
                names_spoken.append(name)

            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (50, 220, 255), cv2.FILLED)
            # cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100, 10), 2)
            draw.text((left + 1, top - 65), name, font=font, fill=(0, 80, 255, 100))
            draw.text((left, top - 66), name, font=font, fill=(255, 255, 255, 10))
            frame = np.array(img_pil)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
tts_d.close()
video_capture.release()
cv2.destroyAllWindows()
