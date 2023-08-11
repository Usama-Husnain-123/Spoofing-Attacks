from flask import Flask , render_template , Response , request  , redirect , url_for , url_for
import cv2
from datetime import datetime, time
import base64
import secrets
import pymongo
from gridfs import GridFS
from bson import ObjectId
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


# create the client of MongoDB 
client = pymongo.MongoClient('mongodb://localhost:27017/')


# Create the DatasetBase in the MongoDB
db = client['dataset']

# Create the Collection in the DataBase of MongoDB
collection = db['information']

# Create the GridFS for storing the image
grid_fs = GridFS(db)

# define the Columns of collection
def collection_columns(currentTime):
    data = {
            "frameID" : None,
            "frameCaptureTime" : currentTime,
            "finalResult" : None,
            "modelServices" : 
                {
                    "s1" : None,
                    "s2" : None,
                    "s3" : None,
                    "s4" : None
                } 
        }
    return data


# Define the Global Variable 
dataID = ""
video_capture = cv2.VideoCapture(0) 


# Generate the  Frames from the WebCam
def generate_frames():

    while video_capture.isOpened():
        success, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def finalizer(dataID):
    res = collection.find_one({'_id': ObjectId(dataID)})
    # Now ge the Services 
    s1 = int(res["modelServices"].get("s1"))
    s2 = int(res["modelServices"].get("s2"))
    s3 = res["modelServices"].get("s3")
    s4 = res["modelServices"].get("s4")
    
    if s1 == None and s2 == None and s3 == None and s4 == None:
        return "Not all Service Run"
    else:
        if s1 == 1 and s2 == 0:
            return "Not Spoofed Image "
        else:
            return "Spoofed Image"


def convert_img_base64(image):
    decoded_data = base64.b64decode(image.split(',')[1])
    image = Image.open(BytesIO(decoded_data))
    image_new = BytesIO()
    image.save(image_new,format='png')
    image_new.seek(0)
    return image_new



# load All the Models 
def load_models():
    printedModel = load_model("Models Files/PrintedImagesModel.h5")
    screenModel  = load_model("Models Files/vgg16_classification-2.h5")
    return printedModel , screenModel


# Create the Object of the Flask
app = Flask(__name__)


# Index Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global dataID
    data = request.get_json()
    image_data = data.get('image_data')
   
    if image_data:
        currentTime = datetime.now().time().isoformat()
        
        # call the Collection columns functions
        data = collection_columns(currentTime)
        
        # Insert the Data in the mongodb
        insertedData = collection.insert_one(data)
        dataID = insertedData.inserted_id

        # Convert the Image into Base64 and put the image in the GridFS
        # image_bytes = base64.b64decode(image_data)
        image_bytes = convert_img_base64(image_data)
        image_id = grid_fs.put(image_bytes, filename="image.png")
        
        # Update the Collection after Adding the Image Frame ID
        collection.update_one({"_id": dataID}, {"$set": {"frameID": image_id}})

        return redirect(url_for('frame_captured'))
    
@app.route('/frame_captured')
def frame_captured():
    res = collection.find_one({'_id': ObjectId(dataID)})
    if res:
        frame_id = res['frameID']
        image_data = grid_fs.get(frame_id).read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image = Image.open(BytesIO(image_data))
        
        # Resize the Image
        imageResize = image.resize((224,224))
        cv2Image    = np.array(imageResize) 
        cv2Image    = cv2Image.astype(dtype = np.float32)

        # Normalize the Image
        cv2Image /= 255

        lambdaService= lambda val: 1 if val > 0.5 else 0
        # load the Models
        printedModel , screenModel = load_models()
        s1  = printedModel.predict(np.expand_dims(cv2Image, axis=0))[0][0]
        s2  = screenModel.predict(np.expand_dims(cv2Image, axis=0))[0][0]
        


        # Update the Values in the DataBase
        #ollection.update_one({"_id": dataID}, {"$set": {"frameID": image_id}})
        collection.update_one(
            {"_id": dataID},
            {"$set": {
                "modelServices.s1": str(lambdaService(s1)),
                "modelServices.s2": str(lambdaService(s2))
            }}
        )

        # call the Finalizer function

        imageInfo = finalizer(dataID)
        
    return render_template("newTemplate.html" , image_data = image_base64 , image_info = imageInfo)



if __name__ == "__main__":
    app.run(port = 9002 , debug = True) 