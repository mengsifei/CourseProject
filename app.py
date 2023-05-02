import os
import io
from flask import Flask, render_template, request, redirect, session, send_file
from werkzeug.utils import secure_filename
from imports import *
import string
from imutils import face_utils
import time
import random

app = Flask(__name__)
app.secret_key = "sifeimeng"
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'wmv', 'avi', 'mkv'}
best_frame = []
is_running = True


def get_best_frame(video_path, model_name, num_frame=0, fps=10, frame_size=800, box_size=300):
    if model_name == "efficientnet":
        model = load_efficientnet()
    else:
        model = load_net()
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # 5 frames per second
    if fps > video_fps or video_fps < fps:
        every_n_frame = 1
    else:
        every_n_frame = video_fps / fps
    if num_frame == 0:
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    is_first = True
    threshold = 0
    best_score, best_nframe, cur_frame, the_best_frame = None, None, None, None
    blur_rate, ear_rate, mar_rate = None, 2, 200
    while is_running:
        ret, frame = cap.read()
        count += 1
        if ret:
            if count % every_n_frame != 0:
                continue
            cur_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect(gray, 0)
            if len(faces) == 0:
                continue
            score_blur = blur_score(gray)
            if is_first:
                blur_rate = score_blur * 2
            shape = predict(gray, faces[0])
            shape = face_utils.shape_to_np(shape)
            not_ok, left_ear = eye_aspect_ratio('left_eye', gray, shape, is_first)
            # print("left ear", left_ear, not_ok)
            if not_ok:
                continue
            not_ok, right_ear = eye_aspect_ratio('right_eye', gray, shape, is_first)
            if not_ok:
                continue
            ear = (left_ear + right_ear) / 2
            opened_mouth, mar = mouth_dist(shape)
            if opened_mouth:
                continue
            # print(mar)
            score_blur /= blur_rate
            mar /= mar_rate
            beauty_score = predict_img(frame, model)
            overall_score = beauty_score + ear + score_blur - mar
            if is_first or overall_score >= threshold:
                threshold = overall_score
                best_nframe = count
                best_score = beauty_score
                is_first = False
                the_best_frame = cur_frame
            if count >= num_frame:
                if best_nframe:
                    break
                else:
                    count = 0
        else:
            break
    if not best_nframe:
        return False, None, None, None, None
    else:
        frame = cv2.resize(the_best_frame, (frame_size, int(the_best_frame.shape[0]
                                                            / (the_best_frame.shape[1] / frame_size))))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect(gray, 0)
        shape = predict(gray, faces[0])
        shape = face_utils.shape_to_np(shape)
        exec_time = time.time() - start_time
        return True, frame, best_score, num_frame / exec_time, exec_time


def is_allowed_file(filename):
    if '.' in filename:
        file_format = filename.rsplit('.')[1]
        if file_format.lower() in ALLOWED_EXTENSIONS:
            return True
    return False


@app.route('/')
def index():
    global is_running
    is_running = True
    return render_template('index.html')


@app.route('/infer', methods=['GET'])
def success():
    num_frames = session.get('num_frames', None)
    frame_size = session.get('frame_size', None)
    filename = session.get('filename', None)
    model_name = session.get('model', None)
    fpsecond = session.get('fps', None)
    if num_frames is None or frame_size is None or filename is None or model_name is None or fpsecond is None:
        return render_template('error.html')
    has_frame, frame, score, fps, exec_time = get_best_frame(os.path.join("uploads", filename), model_name=model_name, num_frame=int(num_frames), fps=int(fpsecond), frame_size=int(frame_size))
    if not is_running:
        return redirect('/delete')
    if not has_frame:
        print("error")
        return render_template('error.html')
    best_frame.append(frame)
    if os.path.isfile(os.path.join("uploads", filename)):
        os.remove(os.path.join("uploads", filename))
    return render_template('inference.html', score=score, image=frame, fps=round(fps, 2), time=round(exec_time, 2))


@app.route('/generate_image')
def generate_image():
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
    global is_running
    is_running = False
    time.sleep(1)
    return redirect('/delete')


@app.route('/delete')
def delete_file():
    filename = session.get('filename', None)
    if os.path.isfile(os.path.join("uploads", filename)):
        os.remove(os.path.join("uploads", filename))
    return redirect('/')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/loading', methods=['GET', 'POST'])
def loading():
    if request.method == 'POST':
        f = request.files['file']
        num_frames = request.form.get('num_frames')
        frame_size = request.form.get('frame_size')
        fps = request.form.get('fps')
        selected_checkboxes = request.form.getlist('checkboxes')
        if len(selected_checkboxes) == 0: # nothing is selected
            session['model'] = "efficientnet"
        else:
            session['model'] = selected_checkboxes[0]
        if num_frames == "":
            num_frames = 0
        if frame_size == "":
            frame_size = 450
        if fps == "":
            fps = 10
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
    app.run(host='0.0.0.0', port=5000, debug=True)

