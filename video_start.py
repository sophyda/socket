from flask import Flask, render_template, Response
import cv2
import jinja2
app = Flask(__name__)

#  VideoCapture可以读取从url、本地视频文件以及本地摄像头的数据
#  camera = cv2.VideoCapture('rtsp://admin:admin@172.21.182.12:554/cam/realmonitor?channel=1&subtype=1')
#  camera = cv2.VideoCapture('test.mp4')
#  0代表的是第一个本地摄像头，如果有多个的话，依次类推
camera = cv2.VideoCapture(1)


def gen_frames():

    while True:
              #  一帧帧循环读取摄像头的数据
            success, frame = camera.read()
            if not success:
                break
            else:
                  #  将每一帧的数据进行编码压缩，存放在memory中
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                  #  使用yield语句，将帧数据作为响应体返回，content-type为image/jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_start')
def  video_start():

      #  通过将一帧帧的图像返回，就达到了看视频的目的。multipart/x-mixed-replace是单次的http请求-响应模式，如果网络中断，会导致视频流异常终止，必须重新连接才能恢复
    return Response(gen_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def  index():

    return render_template('index.html')

if  __name__  ==  '__main__':
        app.run(host='0.0.0.0', debug = True)
