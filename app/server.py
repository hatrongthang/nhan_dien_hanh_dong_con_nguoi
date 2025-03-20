from flask import Flask, render_template, Response
import cv2
from app import ActionDetector
import threading
import time
import telegram
import asyncio
from datetime import datetime

app = Flask(__name__)
detector = ActionDetector()

# Global variables for sharing data between threads
current_frame = None
current_data = {
    'fps': 0,
    'motion_value': 0,
    'current_action': '',
    'confidence': 0,
    'stability': 0,
    'actions_confidence': {},
    'falling_duration': 0  # Thêm biến theo dõi thời gian ngã
}
frame_lock = threading.Lock()
data_lock = threading.Lock()

# Khởi tạo Telegram bot
bot_token = '7879096838:AAFfWOun8NuV0vCNNgY_IAOq7M3LL0wHi84'
chat_id = '-1002528969364'
bot = telegram.Bot(token=bot_token)

# Biến theo dõi trạng thái cảnh báo
falling_start_time = None
falling_alert_sent = False
FALL_ALERT_THRESHOLD = 10  # Thời gian ngã (giây) trước khi gửi cảnh báo

async def send_telegram_alert(frame):
    global falling_alert_sent
    try:
        if not falling_alert_sent:
            # Lưu frame hiện tại
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"fall_detection_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            
            # Tạo message cảnh báo
            message = "⚠️ CẢNH BÁO KHẨN CẤP! ⚠️\n\n"
            message += "🔴 Phát hiện người ngã và nằm quá 10 giây!\n"
            message += f"⏰ Thời gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            
            # Gửi tin nhắn và hình ảnh
            with open(image_path, 'rb') as photo:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=message
                )
            falling_alert_sent = True
            print("Đã gửi cảnh báo qua Telegram!")
    except Exception as e:
        print(f"Lỗi khi gửi cảnh báo Telegram: {str(e)}")

def process_camera():
    global current_frame, current_data, falling_start_time, falling_alert_sent
    rtsp_url = 'rtsp://admin:hoanghue123@172.16.64.251:554/onvif2'
    cap = cv2.VideoCapture(rtsp_url)  # Use webcam
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Detect person and process frame
        bbox = detector.detect_person(frame)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get landmarks for motion calculation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.pose.process(frame_rgb)
            landmarks = []
            if results.pose_landmarks:
                h, w = frame.shape[:2]
                for landmark in results.pose_landmarks.landmark:
                    x_pixel, y_pixel = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x_pixel, y_pixel))
            
            # Predict action
            action, confidence = detector.predict_action(frame, bbox, results, landmarks)
            action, confidence = detector.smooth_prediction(action, confidence)
            action, confidence = detector.get_stable_action(action, confidence, frame.shape, landmarks)
            
            # Xử lý logic theo dõi thời gian ngã
            if action == 'falling' and confidence > detector.confidence_threshold:
                if falling_start_time is None:
                    falling_start_time = current_time
                falling_duration = current_time - falling_start_time
                
                if not falling_alert_sent and falling_duration >= FALL_ALERT_THRESHOLD:
                    # Gửi cảnh báo qua Telegram
                    asyncio.run(send_telegram_alert(frame))
            else:
                falling_start_time = None
                falling_alert_sent = False
                falling_duration = 0
            
            # Calculate motion
            motion = detector.calculate_motion(landmarks, frame.shape)
            
            # Get all confidence scores
            all_confidences = detector.model.predict(detector.preprocess_frame(frame, bbox), verbose=0)[0]
            actions_confidence = {act: float(conf) for act, conf in zip(detector.actions, all_confidences)}
            
            # Calculate stability
            stability = min(len(detector.action_buffer) / detector.buffer_size, 1.0)
            
            # Update current data
            with data_lock:
                current_data.update({
                    'fps': float(fps),
                    'motion_value': float(motion),
                    'current_action': action,
                    'confidence': float(confidence),
                    'stability': float(stability),
                    'actions_confidence': actions_confidence,
                    'falling_duration': falling_duration if falling_start_time is not None else 0
                })
        
        # Update current frame
        with frame_lock:
            current_frame = frame.copy()
        
        time.sleep(0.03)  # Limit processing rate
    
    cap.release()

def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    with data_lock:
        return current_data

if __name__ == '__main__':
    # Start camera processing in a separate thread
    camera_thread = threading.Thread(target=process_camera)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 