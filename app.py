from re import split
from flask import Flask, render_template, redirect, request, Response, url_for
import os
import sqlite3
import cv2
import datetime, time
from tensorflow import keras
import tensorflow
from tensorflow.keras.models import load_model
from flask.helpers import flash

global capture,p
capture = 0
p = ''
cl = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
new_model = load_model('best_model_5L.hdf5')

app.secret_key = "abc526984"


def gen_frames():  # generate frame by frame from camera
    global capture,p
    ca = cv2.VideoCapture(0)
    while True:
        success, frame = ca.read() 
        if success:
               
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                
                cv2.imwrite(p, frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
    ca.release()
    cv2.destroyAllWindows()
    

@app.route('/')
def login():
    return render_template('login.html')
@app.route('/signup', methods = ['GET','POST'] )
def signup():
    error = None
    if request.method == 'POST':
        
        DUN = request.form['rusername']
        DPW = request.form['rpassword']
        CPW = request.form['confirmpassword']
        sqlconnection = sqlite3.Connection(cl + '\login.db')
        cursor = sqlconnection.cursor()
        usr = cursor.execute("SELECT Username FROM Users WHERE Username = '{us}'".format(us = DUN))
        usr = usr.fetchone()
        if usr:
            error = "Username already exist"
        else:
            if DPW == CPW:
                query1 = "INSERT INTO Users VALUES('{u}', '{p}')".format(u = DUN, p = DPW)
                cursor.execute(query1)
                sqlconnection.commit()
                return redirect("/")
            else:
                error = "Password Mismatch"
    return render_template("signup.html",error = error)

@app.route('/tinkerln')
def tinkerln():
    return render_template('tinkerln.html')
@app.route('/verify', methods = ['POST'])
def verify():
    error = None
    if request.method == "POST":
        UN = request.form['username']
        PW = request.form['password']
        sqlconnection = sqlite3.Connection(cl + '\login.db')
        cursor = sqlconnection.cursor()
        query1 = "SELECT Username, Password From Users WHERE Username = '{un}' AND Password = '{pw}'".format(un = UN, pw = PW)
        rows = cursor.execute(query1)
        rows = rows.fetchone()
        if rows:
            
            return render_template("verify.html")
        else:
            error = "Invalid username or password"
            
    
    return render_template("login.html", error = error)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/takeimage', methods=['GET','POST'])
def takeimage():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1             
    elif request.method=='GET':
        return render_template('camera.html')
    return render_template('camera.html')
    
    # if request.method == "POST":

    #     ca = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
            
    #     success, frame = ca.read()
        
    #     now = datetime.datetime.now()
    #     p = './shots/file2.jpg'
    #     cv2.imwrite(p, frame)
            
    #     ca.release()
    #     cv2.destroyAllWindows()
            
        
    # return render_template("camera.html")

@app.route('/predict', methods=['POST'])
def predict():
    global p
    error = None
    if request.method=='POST':
        
        import numpy as np
        categories = ['1','2']
        
        # Read the input image
        img = cv2.imread('{}'.format(p))
        
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangle around the faces and crop the faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = img[y:y + h, x:x + w]
            
        print(faces.shape)
        faces = cv2.resize(faces, dsize = (227,227))
        print(faces.shape)

        test_image = np.array([faces], dtype=np.float16) / 255.0
        result = new_model.predict(test_image)
        result = list(result)
        print(result)
        print(categories[np.argmax(result)])
        if categories[np.argmax(result)] == '2':
            return render_template('chat.html')
        else:
            error = "Oops!!!....You are not authorised to use this website"
            return render_template('camera.html',error = error)
    return render_template('camera.html')



if __name__ == "__main__":
    app.run()
