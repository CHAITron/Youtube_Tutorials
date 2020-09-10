import speech_recognition as sr  #ใช้สำหรับแปลงเสียงพูดเป็นข้อความ
from gtts import gTTS #ใช้สำหรับแปลงข้อความเป็นเสียงพูด
from playsound import playsound #ใช้สำหรับเล่นไฟล์เสียง
from datetime import datetime #ใช้สำหรับดูเวลาขณะนี้


r = sr.Recognizer() #เริ่มต้นInitiate

with sr.Microphone() as source: 
	playsound("./signal.mp3") #ส่งสัญญาณเตือน
	audio = r.record(source, duration=5) #บันทึกเสียง 5 วินาที
	playsound("./signal.mp3") #ส่งสัญญาณเตือน

	try:
		text = r.recognize_google(audio, language="th") #ส่งไปให้google cloud
		if "ผม" in text:
			text = text.replace("ผม", "ฉันเองก็")
		if "ครับ" in text:
			text = text.replace("ครับ", "ค่ะ")
		if text == "กี่โมงแล้ว":
			now = datetime.now() #รับค่าเวลาขณะนั้น
			text = now.strftime("ขณะนี้เวลา%Hนาฬิกา%Mนาที%Sวินาที")
	except:
		text = "ขอโทษค่ะ"
	tts = gTTS(text, lang="th") #ส่งไปให้google cloud
	tts.save("./answer.mp3") #บันทึกเสียงที่ได้จากgoogle cloud
	playsound("./answer.mp3")
