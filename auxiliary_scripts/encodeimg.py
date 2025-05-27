import cv2 # type: ignore
import face_recognition # type: ignore
 
imgPsh = face_recognition.load_image_file('/home/siddharth/Downloads/mlpbl/peh/Prashnat.jpeg')
imgPsh = cv2.cvtColor(imgPsh,cv2.COLOR_BGR2RGB)
imgSid = face_recognition.load_image_file('/home/siddharth/Downloads/mlpbl/peh/Siddharth.jpg')
imgSid = cv2.cvtColor(imgSid,cv2.COLOR_BGR2RGB)
imgKan = face_recognition.load_image_file('/home/siddharth/Downloads/mlpbl/peh/Kanak.jpeg')
imgKan = cv2.cvtColor(imgKan,cv2.COLOR_BGR2RGB)
 
facePsh = face_recognition.face_locations(imgPsh)[0]
encodePsh = face_recognition.face_encodings(imgPsh)[0]
cv2.rectangle(imgPsh,(facePsh[3],facePsh[0]),(facePsh[1],facePsh[2]),(255,0,255),2)
 
faceSid = face_recognition.face_locations(imgSid)[0]
encodeSid = face_recognition.face_encodings(imgSid)[0]
cv2.rectangle(imgSid,(faceSid[3],faceSid[0]),(faceSid[1],faceSid[2]),(255,0,255),2)

faceKan = face_recognition.face_locations(imgKan)[0]
encodeKan = face_recognition.face_encodings(imgKan)[0]
cv2.rectangle(imgKan,(faceKan[3],faceKan[0]),(faceKan[1],faceKan[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodePsh],encodeKan)
faceDis = face_recognition.face_distance([encodePsh],encodeKan)

print(results,faceDis)
cv2.putText(imgKan,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Prasnnat',imgPsh)
cv2.imshow('Siddharth',imgSid)
cv2.imshow('Kanak',imgKan)

cv2.waitKey(0)

