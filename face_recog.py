import face_recognition as face 
import numpy as np 
import cv2

#ORIGINAL_CODE_CREDIT:  https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

#ดึงวิดีโอตัวอย่างเข้ามา, ถ้าต้องการใช้webcamให้ใส่เป็น0
video_capture = cv2.VideoCapture("sample.mp4") 

#ใบหน้าคนที่ต้องการรู้จำเป็นreference #คนที่1
pop_image = face.load_image_file("pop.jpg")
pop_face_encoding = face.face_encodings(pop_image)[0]

#ใบหน้าคนที่ต้องการรู้จำเป็นreference #คนที่2
oat_image = face.load_image_file("oat.jpg")
oat_face_encoding = face.face_encodings(oat_image)[0]

#ประกาศตัวแปร
face_locations = []
face_encodings = []
face_names = []
face_percent = []
#ตัวแปรนี้ใช้สำหรับคิดเฟรมเว้นเฟรมเพื่อเพิ่มfps 
process_this_frame = True

known_face_encodings = [pop_face_encoding, oat_face_encoding]
known_face_names = ["PONGKUL", "PRAMOTE"]

#loopคำนวณแต่ละเฟรมของวิดีโอ
while True:
    #อ่านค่าแต่ละเฟรมจากวิดีโอ
    ret, frame = video_capture.read()
    if ret:
        #ลดขนาดสองเท่าเพื่อเพิ่มfps 
        small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
        #เปลี่ยน bgrเป็น rgb 
        rgb_small_frame = small_frame[:,:,::-1]

        face_names = []
        face_percent = []

        if process_this_frame:
            #ค้นหาตำแหน่งใบหน้าในเฟรม 
            face_locations = face.face_locations(rgb_small_frame, model="cnn")
            #นำใบหน้ามาหาfeaturesต่างๆที่เป็นเอกลักษณ์ 
            face_encodings = face.face_encodings(rgb_small_frame, face_locations)
            
            #เทียบแต่ละใบหน้า
            for face_encoding in face_encodings:
                face_distances = face.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_distances)
                face_percent_value = 1-face_distances[best]

                #กรองใบหน้าที่ความมั่นใจ50% ปล.สามารถลองเปลี่ยนได้
                if face_percent_value >= 0.5:
                    name = known_face_names[best]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                else:
                    name = "UNKNOWN"
                    face_percent.append(0)
                face_names.append(name)

        #วาดกล่องและtextเมื่อแสดงผลออกมาออกมา
        for (top,right,bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top*= 2
            right*= 2
            bottom*= 2
            left*= 2

            if name == "UNKNOWN":
                color = [46,2,209]
            else:
                color = [255,102,51]

            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left-1, top -30), (right+1,top), color, cv2.FILLED)
            cv2.rectangle(frame, (left-1, bottom), (right+1,bottom+30), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)


        #สลับค่าเป็นค่าตรงข้ามเพื่อให้คิดเฟรมเว้นเฟรม
        process_this_frame = not process_this_frame

        #แสดงผลลัพท์ออกมา
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

#ล้างค่าต่างๆเมื่อปิดโปรแกรม
video_capture.release()
cv2.destroyAllwindow()
