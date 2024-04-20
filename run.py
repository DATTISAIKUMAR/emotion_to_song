import cv2
import webview
from flask import Flask, render_template, request, url_for,send_from_directory,Response
import sqlite3
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pyautogui as py
import os
import time
import random
import pandas as pd







with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()


loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model.load_weights("model_weights.weights.h5")




app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

webview.create_window("hello",app)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

















list1=["Anger","Contempt","Dark","Disgust",'Fear',"Happy","Nautral","Sad","Surprised"]









Anger=["  aadevadanna edevedanna","  nippu ra","  dorikithe chastavu","  rage of narappa"]
Contempt=["  mr.perfect-DSP Mix","  satte era satte","  Magadheera(from oke okkadu)","  varam nan Unai"]
Disgust=["  chi chi cHi","  gaajuvaka Pilla","  Aa ante","  choopultho Guchi"]
Fear=["  I'm scared","  nandikonda","  vilaya pralaya Moorthy","  beggin"]
Happy=["  Tillu anna Dj pedithe","  hoyna hoyna","  gallo Thelinattunde","  Gundellonaa"]
Neutral=["  oke okka jeevitham","   manusu maree","  chiru chiru","  telusa manasa"]
Sad=["  gelupu thalupu","  adiga adiga","  chenchala","  nee prashnalu"]
Surprised=["  aashiqui2 mashup","  kaanunna kalyanam","  nennu nuvvantu","   inka edho"]


dict={0:Anger,1:Contempt,3:Disgust,4:Fear,5:Happy,6:Neutral,7:Sad,8:Surprised}
'''dict1={"Anger":Anger,"Contempt":Contempt,"Disgust":Disgust,"Fear":Fear,"Happy":Happy,"Neutral":Neutral,"Sad":Sad,"Surprised":Surprised}
dataframe=pd.DataFrame(dict1)
print(dataframe)
'''








@app.route('/')
def home():
    return render_template('login.html')
@app.route('/start')
def start():
    return render_template("main_page.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['psw']
        psw1 = request.form['psw1']
        if(username !="" and password!="" and password==psw1):
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('Select * from users where email=(?) and password=(?)',(username,password))
            data=cursor.fetchone()
            if(data):
                msg="user already exists please login"
                return render_template('register.html',msg=msg)
            else:
                if(not data):
                    cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (username, password))
                    conn.commit()
                    conn.close()
                msg="successfully login"
                return render_template('login.html',msg=msg)
        else:
            msg="invalid values"
            return render_template("register.html",msg=msg)

    else:
        msg = "request to register page"
        return render_template('register.html',msg=msg)






@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form['email1']
        password = request.form['psw1']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('Select * from users where email=(?) and password=(?)', (username, password))
        d = cursor.fetchone()
        if(d):
            return render_template('start.html')
        else:
            msg="invalid password or invalid username"
            return render_template('login.html',msg=msg)
    else:
        msg = "login page request"
        return render_template("login.html",msg=msg)






def re_size(filepath):
    img = image.load_img(filepath,target_size=(250,250))
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values between 0 and 1
    val1 = loaded_model.predict(img_array)  # Assuming loaded_model is defined elsewhere
    val1 = np.argmax(val1)
    return val1


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            global prediction
            img_path="uploads\\"+filename
            prediction=re_size(img_path)
            return render_template('file.html', prediction=list1[prediction],file_url=file_url)

    return render_template('file.html', prediction=None,file_url=None)





def get_frame():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        img2 = cv2.resize(frame, (250, 250))
        img2 = np.expand_dims(img2, axis=0)
        val1 = loaded_model.predict(img2)
        val1 = np.argmax(val1)
        global prediction
        prediction = val1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 0, 0), 5)
            cv2.putText(frame, list1[val1], (h + 14, w + 14), cv2.FONT_HERSHEY_DUPLEX, 0.7, (250, 0, 0), 1, cv2.LINE_AA)


        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/camera')
def videos():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/play")
def play_song():

    if (prediction == 2):
        msg = "you are in dark place emotion can not find out"
        return render_template("file.html", msg=msg)

    os.system("spotify")
    time.sleep(10)
    py.hotkey('ctrl', 'l')
    py.write(dict[prediction][random.randint(0, 3)], interval=0.1)
    for i in ['enter', 'pagedown', 'tab', 'enter', 'enter']:
        time.sleep(2)
        py.press(i)
    return render_template("start.html")



@app.route('/file_page')
def file_page():
    return render_template("main_page.html")

@app.route("/main")
def main_pg():
    return render_template("file.html")





if __name__ == '__main__':
    webview.start()