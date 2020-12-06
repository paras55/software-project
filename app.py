from flask import Flask, render_template, request
from flask import Flask, request, render_template, redirect, url_for
from flask import Flask, render_template, request,send_file
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from google.colab.patches import cv2_imshow

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model = load_model("mask_recog.h5")


app = Flask(__name__)


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/login')
def login():
   return render_template('login.html')

@app.route('/upload')
def upload():
   return render_template('upload.html')

@app.route('/query',methods=["POST", "GET"])
def query():
    print('hello')
    if request.method=="POST": 
        username = request.form['username']
        password = request.form['password']
        if username == 'thapar' and password == 'thapar':
            return render_template('admin.html')
    else:
            return render_template('login.html')
def face_mask_detector(frame):
  # frame = cv2.imread(fileName)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
  faces_list=[]
  preds=[]
  for (x, y, w, h) in faces:
      face_frame = frame[y:y+h,x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))
      face_frame = img_to_array(face_frame)
      face_frame = np.expand_dims(face_frame, axis=0)
      face_frame =  preprocess_input(face_frame)
      faces_list.append(face_frame)
      if len(faces_list)>0:
          preds = model.predict(faces_list)
      for pred in preds:
          (mask, withoutMask) = pred
      label = "Mask" if mask > withoutMask else "No Mask"
      color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
      cv2.putText(frame, label, (x, y- 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
  # cv2_imshow(frame)
  return frame

@app.route('/image', methods = ['GET', 'POST'])
def upload_image():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      input_image = cv2.imread(f.filename)
      output = face_mask_detector(input_image)
      filename = 'hello.jpg'
      cv2.imwrite(filename, output) 
      return send_file(filename, as_attachment=True)
      #return cv2_imshow(output)

@app.route('/video', methods = ['GET', 'POST'])
def upload_video():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      cap = cv2.VideoCapture(f.filename)
      ret, frame = cap.read()
      frame_height, frame_width, _ = frame.shape
      out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
      print("Processing Video...")
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          out.release()
          break
        output = face_mask_detector(frame)
        out.write(output)
      out.release()
      return send_file('output.mp4', as_attachment=True)
		
if __name__ == '__main__':
   app.run()

