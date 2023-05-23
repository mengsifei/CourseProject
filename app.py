import os
import io
from flask import Flask, render_template, request, redirect, session, send_file
from werkzeug.utils import secure_filename
from src.net import *
from src.primary_check import *
from src.secondary_check import *
import string
from imutils import face_utils
import time
import random

app = Flask(__name__)
app.secret_key = "sifeimeng"
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'wmv', 'avi', 'mkv'}
best_frame = []
is_running = True


def frame_resize(frame, frame_size=450):
    """
    :param frame: the image to be resized
    :param frame_size: the width of the resized image, by default it equals to 450 pixels.
    :return: resized frame
    """
    return cv2.resize(frame, (frame_size, int(frame.shape[0] / (frame.shape[1] / frame_size))))


def get_best_frame(video_path, num_frame=0, fps=3, frame_size=450, debug=False):
    """
    :param video_path: the path of the video. The filename is a string of 8 random symbols
    :param num_frame: the first number of frames of a video to be analyzed
    :param fps: the number of frames to be analyzed in a second
    :param frame_size: the width of the final frame
    :param debug: boolean value that will print values when equals to True
    :return:
     1. Boolean value if a suitable frame is found
     2. The best frame
     3. The execution time of the algorithm
    """
    if not os.path.isfile(video_path):
        return False, None, None
    # initialization
    model = load_efficientnet()
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    every_n_frame = max(1, round(video_fps / fps))
    num_frame, fps = abs(num_frame), abs(fps)
    if num_frame == 0 or num_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count, is_first, spare_eye, spare_mouth = 0, True, None, None
    the_best_frame = {"score": None, "frame": None, "threshold": None, "count": None}
    calibrations = {"ear": None, "blur": None, "mar": None, 'beauty': None}
    normalizations = {"ear": 1, "blur": 1, "mar": 30, "beauty": None}
    while is_running:
        ret, frame = cap.read()
        count += 1
        if ret:
            # analyze n frames per second
            if count % every_n_frame != 0:
                continue
            # face detection
            frame = frame_resize(frame, 450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect(gray, 0)
            # if no faces detected
            if len(faces) == 0:
                continue
            # facial landmarks detection
            face = faces[0]
            shape = predict(gray, face)
            shape = face_utils.shape_to_np(shape)
            # count ear
            closed_eyes_l, left_ear = eye_aspect_ratio('left_eye', shape)
            closed_eyes_r, right_ear = eye_aspect_ratio('right_eye', shape)
            if closed_eyes_l or closed_eyes_r:
                continue
            # ear score
            if is_first:
                calibrations['ear'] = round((left_ear + right_ear) / 2, 4)
                normalizations['ear'] = 2 / (left_ear + right_ear)
            ear_score = min((left_ear + right_ear - calibrations['ear']) * normalizations['ear'], 1)
            ear_score = max(ear_score, -1)
            # open mouth detection
            opened_mouth, mar = mouth_dist(shape)
            if is_first or mar < 0.1:
                spare_eye = frame
            if opened_mouth:
                continue
            if is_first:
                spare_mouth = frame
                calibrations['mar'] = round(mar, 4)
            mar_score = (mar - calibrations['mar']) * normalizations['mar']
            mar_score = min(mar_score, 1)
            mar_score = max(mar_score, -1)
            # gaze detection
            left_gaze = looking_center(gray, shape, "left_eye")
            right_gaze = looking_center(gray, shape, "right_eye")
            if left_gaze == -2 or right_gaze == -2:
                continue
            look_center_score = (left_gaze + right_gaze) / 2
            blur_score = detect_blur(gray)
            if is_first:
                calibrations['blur'] = int(blur_score)
            blur_score = blur_score / calibrations['blur']
            if blur_score < 1:
                blur_score = -1
            blur_score = min(blur_score, 1)
            blur_score = max(blur_score, -1)
            beauty_score = predict_img(frame, model)
            if is_first:
                calibrations['beauty'] = int(beauty_score)
                normalizations['beauty'] = 1 / (beauty_score - calibrations['beauty'])
            beauty_score = (beauty_score - calibrations['beauty']) * normalizations['beauty']
            beauty_score = min(beauty_score, 1)
            beauty_score = max(beauty_score, -1)
            # count general score
            overall_score = beauty_score + ear_score + look_center_score + blur_score - mar_score
            if debug:
                print("count", count, "overall", overall_score, "look score", look_center_score, "beauty", beauty_score,
                      "mar", mar_score, "ear", ear_score, "blur", blur_score)
            # update best_frame
            if is_first or overall_score >= the_best_frame['threshold']:
                the_best_frame['threshold'] = overall_score
                the_best_frame['score'] = overall_score
                the_best_frame['frame'] = frame
                the_best_frame['count'] = count
                is_first = False
            if count >= num_frame:
                if the_best_frame['count']:
                    break
                else:
                    count = 0
                    continue
        else:
            break
    if the_best_frame['count'] is None and spare_eye is None:
        return False, None, None
    elif the_best_frame['count'] is None and spare_eye is not None:
        if spare_mouth is not None:
            frame = frame_resize(spare_mouth, frame_size)
        else:
            frame = frame_resize(spare_eye, frame_size)
        exec_time = time.time() - start_time
        return True, frame, exec_time
    else:
        frame = frame_resize(the_best_frame['frame'], frame_size)
        exec_time = time.time() - start_time
        return True, frame, exec_time


def is_allowed_file(filename):
    """
    :param filename: the filename of the video (string)
    :return: boolean value if the format is allowed
    """
    if '.' in filename:
        file_format = filename.rsplit('.')[1]
        if file_format.lower() in ALLOWED_EXTENSIONS:
            return True
    return False


@app.route('/')
def index():
    """
    the home page of the web-app
    """
    global is_running
    is_running = True
    return render_template('index.html')


@app.route('/infer', methods=['GET'])
def success():
    """
    Runs the algorithm.
    """
    num_frames = session.get('num_frames', None)
    frame_size = session.get('frame_size', None)
    filename = session.get('filename', None)
    fpsecond = session.get('fps', None)
    if num_frames is None or frame_size is None or filename is None or fpsecond is None:
        return render_template('error.html')
    has_frame, frame, exec_time = get_best_frame(os.path.join("uploads", filename), num_frame=int(num_frames),
                                                 fps=int(fpsecond), frame_size=int(frame_size))
    if not is_running:
        # if interrupted
        return redirect('/delete')
    if not has_frame:
        # if no frame generated
        return render_template('error.html')
    best_frame.append(frame)
    if os.path.isfile(os.path.join("uploads", filename)):
        # delete uploaded file
        os.remove(os.path.join("uploads", filename))
    return render_template('inference.html', image=frame, time=round(exec_time, 2))


@app.route('/generate_image')
def generate_image():
    """
    Send the byte format of the best frame
    """
    filename = session.get('filename', None)
    filename = filename.rsplit('.')[0]
    img = best_frame[-1]
    ret, encoded_image = cv2.imencode('.jpg', img)
    if not ret:
        return render_template('error.html')
    file = io.BytesIO(encoded_image)
    return send_file(file, mimetype='image/jpg', as_attachment=True,
                     download_name="best_frame_{}.jpg".format(filename))


@app.route('/interrupt_execution')
def interrupt():
    """
    Interrupts execution and deletes file
    """
    global is_running
    is_running = False
    time.sleep(1)
    return redirect('/delete')


@app.route('/delete')
def delete_file():
    """
    Delete file if filename is detected
    """
    filename = session.get('filename', None)
    if os.path.isfile(os.path.join("uploads", filename)):
        os.remove(os.path.join("uploads", filename))
    return redirect('/')


@app.route('/about')
def about():
    """
    Redirect to Documentation page
    """
    return render_template('about.html')


@app.route('/loading', methods=['GET', 'POST'])
def loading():
    """
    Loading page when the algorithm is running. The file submitted from home page will be saved to session.
    The filename will be automatically generated.
    """
    if request.method == 'POST':
        f = request.files['file']
        num_frames = request.form.get('num_frames')
        frame_size = request.form.get('frame_size')
        fps = request.form.get('fps')
        if num_frames == "":
            num_frames = 0
        if frame_size == "":
            frame_size = 450
        if fps == "":
            fps = 3
        session['num_frames'] = num_frames
        session['frame_size'] = frame_size
        session['fps'] = fps
        if f.filename == "":
            return redirect('/')
        if is_allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_format = filename.rsplit('.')[1].lower()
            new_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + '.' + file_format
            f.save(os.path.join("uploads", new_filename))
            session['filename'] = new_filename
        else:
            return redirect('/')
    return render_template('loading.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
