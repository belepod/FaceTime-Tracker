import cv2
import numpy as np
import face_recognition
import os
import datetime

path = '/home/siddharth/Downloads/mlpbl/peh/img'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

siddharth_image_path = next((f'{path}/{cl}' for cl in myList if cl.startswith('Siddharth')), None)
if siddharth_image_path:
    siddharth_img = cv2.imread(siddharth_image_path)
    encodeListKnown = findEncodings([siddharth_img])
else:
    raise FileNotFoundError("Siddharth's image not found in the specified directory.")

print('Encoding Complete')

attendance_status = {}

def markAttendance(name, face_present, current_time=None):
    if current_time is None:
        current_time = datetime.datetime.now()

    if name not in attendance_status:
        attendance_status[name] = {
            "present": False,
            "last_seen": current_time,
            "exit_time": None
        }

    if face_present:
        if not attendance_status[name]["present"]:
            entry_time = current_time.strftime('%H:%M:%S')

            with open('log.csv', 'a') as f:
                f.writelines(f'\n{name},Entered,{entry_time}')

            attendance_status[name]["present"] = True
            attendance_status[name]["last_seen"] = current_time

    else:
        if attendance_status[name]["present"]:
            exit_time = current_time.strftime('%H:%M:%S')

            with open('log.csv', 'a') as f:
                f.writelines(f'\n{name},Exited,{exit_time}')

            attendance_status[name]["present"] = False
            attendance_status[name]["exit_time"] = current_time

cap = cv2.VideoCapture(0)
last_recorded_times = {}

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    current_time = datetime.datetime.now()

    detected_names = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        name = "UNKNOWN"

        if True in matches:
            name = "SIDDHARTH"

        detected_names.append(name)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        markAttendance(name, face_present=True, current_time=current_time)

        last_recorded_times[name] = current_time

    for name in attendance_status:
        if name not in detected_names:
            if name in last_recorded_times:
                time_since_last_seen = current_time - last_recorded_times[name]
                if time_since_last_seen.total_seconds() > 3:
                    markAttendance(name, face_present=False, current_time=current_time)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
