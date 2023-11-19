from flask import Flask, render_template, request, redirect, url_for , session
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
import imutils


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgThre)

    contours, heirarchy = cv2.findContours(
        imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalCountours

def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warpTime(img,points,w,h,pad=20):
    points = reorder(points)
    pts1 =np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad: imgWarp. shape[1]-pad]
    return imgWarp

def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2+(pts2[1]-pts1[1])**2)**0.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blend')
def blend():
    return redirect('/number_page')

@app.route('/dist')
def dist():
    return render_template('upload_file.html')

@app.route('/detect')
def detect():
    return render_template('upload_file2.html')

@app.route('/number_page')
def number_page():
    return render_template('number_page.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        image_count = int(request.form.get('imageCount', 0))
        session['variable'] = int(image_count)
        return render_template('upload_page.html', image_count=image_count)
    return redirect('/')

@app.route('/upload1', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_image_path = 'temp_image.png'
        uploaded_file.save(uploaded_image_path)
        scale = 3
        wp = 210*scale
        hp = 297*scale
        image = cv2.imread(uploaded_image_path)
        imageContours, conts = getContours(
            image, minArea=50000, filter=4)
        if len(conts) != 0:
            biggest = conts[0][2]
            imgWarp = warpTime(image, biggest, wp, hp)
            imgContours2, conts2 = getContours(imgWarp,
                                                    minArea=2000, filter=4,
                                                    cThr=[50, 50])
            if len(conts) != 0:
                for obj in conts2:
                    cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                    nPoints = reorder(obj[2])
                    nW = round((findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10),1)
                    nH = round(
                        (findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10), 1)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                                                    (0, 0, 255), 3, 8, 0, 0.05)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                                                                    (0, 0, 255), 3, 8, 0, 0.05)
                    x, y, w, h=obj[3]
                    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,
                                (0, 0, 0), 2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 40, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,
                                (0, 0, 0), 2)

        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        cv2.imwrite(processed_image_path, imgContours2)
        res_img = Image.open(processed_image_path)
        result_image_base64 = image_to_base64(res_img)
        uploaded_image = Image.open(uploaded_image_path)
        uploaded_image_base64 = image_to_base64(uploaded_image)
        os.remove(uploaded_image_path)
        return render_template('result.html', result_image=result_image_base64 , uploaded_image=uploaded_image_base64)
    
@app.route('/upload2', methods=['POST'])
def upload_file2():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_image_path = 'temp_image.png'
        uploaded_file.save(uploaded_image_path)
        config_path = 'yolov3.cfg'
        weights_path = 'yolov3.weights'
        classes_path = 'yolov3.txt'

        def get_output_layers(net):
            layer_names = net.getLayerNames()
            try:
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            except:
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            return output_layers

        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            label = str(classes[class_id])
            color = COLORS[class_id]
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
            cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        image = cv2.imread(uploaded_image_path)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(weights_path, config_path)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))


        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        cv2.imwrite(processed_image_path, image)
        res_img = Image.open(processed_image_path)
        result_image_base64 = image_to_base64(res_img)
        uploaded_image = Image.open(uploaded_image_path)
        uploaded_image_base64 = image_to_base64(uploaded_image)
        os.remove(uploaded_image_path)
        return render_template('result.html', result_image=result_image_base64 , uploaded_image=uploaded_image_base64)
    
@app.route('/process', methods=['POST'])
def process():
    try:
        image_count = session.get('variable', None)
        print("Image Count:", image_count)

        uploaded_images = []

        for i in range(image_count):
            file_key = f'image_{i + 1}'
            if file_key in request.files:
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    uploaded_images.append(file_path)
        encoded=[]
        for i in range(len(uploaded_images)):
            uploaded_image = Image.open(uploaded_images[i])
            uploaded_image_base64 = image_to_base64(uploaded_image)
            encoded.append(uploaded_image_base64)

        images = []
        for image_path in uploaded_images:
            img = cv2.imread(image_path)
            images.append(img)

        imageStitcher = cv2.Stitcher_create()
        error, stitched_img = imageStitcher.stitch(images)

        if not error:
            stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
            gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
            thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            areaOI = max(contours, key=cv2.contourArea)
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            x, y, w, h = cv2.boundingRect(areaOI)
            cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

            minRectangle = mask.copy()
            sub = mask.copy()

            while cv2.countNonZero(sub) > 0:
                minRectangle = cv2.erode(minRectangle, None)
                sub = cv2.subtract(minRectangle, thresh_img)
            contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            areaOI = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(areaOI)
            stitched_img = stitched_img[y:y + h, x:x + w]
        else:
            print("Images could not be stitched!")
            print("Likely not enough keypoints being detected!")
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stitchedOutputProcessed.png')
        cv2.imwrite(processed_image_path, stitched_img)
        res_img = Image.open(processed_image_path)
        result_image_base64 = image_to_base64(res_img)
        return render_template('display_images.html',result_img=result_image_base64, images=encoded)
    except Exception as e:
        print("Error:", str(e))
        return "An error occurred."

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
