import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import requests

###########################################

img_width, img_height = 150,150
model_path = 'C:/Users/dell/Desktop/AIrizal/models/model.h5'
model_weights_path = 'C:/Users/dell/Desktop/AIrizal/models/weights.h5'
model = load_model(model_path)
#model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'png', 'PNG'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

#############################################


def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)

    
    if answer == 0:
      print("Label: Calinawan Cave")
      print("Name: https://wonderfilledjournal.com/2016-tanay-rizal-calinawan-cave/")
    elif answer == 1:
      print("Label: Daraitan")
      print("Name: https://wonderfilledjournal.com/2016-tanay-rizal-daranak-falls-batlag-falls/")
    elif answer == 2:
      print("Label: Daranak")
      print("Name: https://wonderfilledjournal.com/2016-tanay-rizal-daranak-falls-batlag-falls/")
    elif answer == 3:
      print("Label: Hapunang Banoi")
      print("Name: Ito ay ang Hapunang Banoi sa Rizal")
    elif answer == 4:
      print("Label: Kinabuan Falls")
      print("Name: Ito ay ang Kinabuan Falls sa Rizal")
    elif answer == 5:
      print("Label: Mt. Lagyo")
      print("Name: Ito ay ang Mt. Lagyo sa Rizal")
    elif answer == 6:
      print("Label: Mt. Oro")
      print("Name: Ito ay ang Mt. Oro sa Rizal")
    elif answer == 7:
      print("Label: Mystical Cave")
      print("Name: Ito ay ang Mystical Cave sa Riza")
    elif answer == 8:
      print("Label: Palo Alto")
      print("Name: Ito ay ang Palo Alto sa Rizal")
    elif answer == 9:
      print("Label: Tungtong Falls")
      print("Name: https://www.celineism.com/2016/05/tongtong-falls-tanay-rizal-photos.html")
    elif answer == 10:
      print("Label: Wawa dam")
      print("Name: https://www.wander.am/travel/manila-74/places/wawa-dam-80919.en.html")

    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='',name='', imagesource='../uploads/template.jpg')


@app.route('/', methods=['GET', 'POST'])

def upload_file():
 
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
              label= 'Calinawan Cave'
              name= 'https://wonderfilledjournal.com/2016-tanay-rizal-calinawan-cave/'
            elif result == 1:
              label= 'Daraitan'
              name= 'https://www.celineism.com/2014/06/ive-seen-omega-at-mount-daraitan-tanay.html'
            elif result == 2:
              label='Daranak'
              name='https://wonderfilledjournal.com/2016-tanay-rizal-daranak-falls-batlag-falls/'
            elif result == 3:
              label='Hapunang Banoi'
              name='Ito ay ang Hapunang Banoi sa Rizal'
            elif result == 4:
              label='Kinabuan Falls'
              name='Ito ay ang Kinabuan Falss sa Rizal'
            elif result == 5:
              label= 'Mt. Lagyo'
              name= 'Ito ay ang Mt. Lagyo sa Rizal'
            elif result == 6:
              label='Mt. Oro'
              name='Ito ay ang Mt. Oro sa Rizal'
            elif result == 7:
              label= 'Mystical Cave'
              name= 'Ito ay ang Mystical Cave sa Rizal'
            elif result == 8:
              label= 'Palo Alto'
              name= 'Ito ay ang Palo Alto sa Rizal'
            elif result == 9:
              label= 'Tungtong Falls'
              name= 'https://www.celineism.com/2016/05/tongtong-falls-tanay-rizal-photos.html'
            elif result == 10:
              label= 'Wawa dam'
              name= 'https://www.wander.am/travel/manila-74/places/wawa-dam-80919.en.html'

            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label,name=name, imagesource='../uploads/' + filename)


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

####################################### SEARCH

@app.route('/', methods=[ 'GET' 'POST'])

def search():
    label = [
{
	"name":"CALINAWAN",
    "url":"https://wonderfilledjournal.com/2016-tanay-rizal-calinawan-cave/"
},
{
	"name":"DARAITAN",
    "url":"https://wonderfilledjournal.com/2016-tanay-rizal-calinawan-cave/"
},
{
	"name":"DARANAK",
    "url":"https://wonderfilledjournal.com/2016-tanay-rizal-calinawan-cave/"
}
]

    return render_template('template.html', label=label)




if __name__ == "__main__":
    app.debug = False
    app.run('127.0.0.1',5400)

