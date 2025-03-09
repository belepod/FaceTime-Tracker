import cv2  # type: ignore
import face_recognition  # type: ignore

# Load and process the images
imgSid = face_recognition.load_image_file('/home/siddharth/Downloads/mlpbl/peh/Siddharth.jpg')
imgSid = cv2.cvtColor(imgSid, cv2.COLOR_BGR2RGB)

# Extract the face encoding for Siddharth
faceSid = face_recognition.face_locations(imgSid)[0]
encodeSid = face_recognition.face_encodings(imgSid)[0]
cv2.rectangle(imgSid, (faceSid[3], faceSid[0]), (faceSid[1], faceSid[2]), (255, 0, 255), 2)

# Load and process the image to be tested
imgKan = face_recognition.load_image_file('/home/siddharth/Downloads/mlpbl/peh/Kanak.jpeg')
imgKan = cv2.cvtColor(imgKan, cv2.COLOR_BGR2RGB)

# Extract the face encoding for the test image
faceKan = face_recognition.face_locations(imgKan)[0]
encodeKan = face_recognition.face_encodings(imgKan)[0]
cv2.rectangle(imgKan, (faceKan[3], faceKan[0]), (faceKan[1], faceKan[2]), (255, 0, 255), 2)

# Compare the test face with Siddharth's face encoding
results = face_recognition.compare_faces([encodeSid], encodeKan)
faceDis = face_recognition.face_distance([encodeSid], encodeKan)

# Determine the name to display
if results[0]:
    name = "SIDDHARTH"
else:
    name = "UNKNOWN"

print(results, faceDis)
cv2.putText(imgKan, f'{name} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Show the images
cv2.imshow('Siddharth', imgSid)
cv2.imshow('Kanak', imgKan)

cv2.waitKey(0)
