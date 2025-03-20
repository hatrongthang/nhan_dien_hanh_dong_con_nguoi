import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import telegram
import asyncio
from datetime import datetime

class ActionDetector:
    def __init__(self):
        # Load model đã train
        self.model = load_model('models/best_model_3.h5')
        
        # Lấy kích thước input từ model
        self.input_shape = self.model.input_shape[1:3]
        print(f"Model input shape: {self.input_shape}")
        
        # Định nghĩa các classes
        self.actions = ['falling', 'jumping', 'running', 'sitting', 'standing', 'walking']
        
        # Khởi tạo MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Tăng độ tin cậy phát hiện pose
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.confidence_threshold = 0.8  # Tăng ngưỡng tin cậy
        self.smooth_factor = 0.3  # Giảm để ổn định hơn
        self.prev_action = None
        self.action_history = []
        self.history_size = 10  # Tăng kích thước history
        self.prev_landmarks = None
        self.motion_threshold = 0.02  # Điều chỉnh ngưỡng chuyển động
        self.action_buffer = []
        self.buffer_size = 15  # Tăng buffer size để ổn định hơn
        self.min_action_duration = 0.5  # Tăng thời gian tối thiểu giữa các hành động
        self.last_action_time = time.time()
        self.vertical_motion_threshold = 0.03  # Ngưỡng chuyển động theo chiều dọc
        self.horizontal_motion_threshold = 0.02  # Ngưỡng chuyển động theo chiều ngang
        self.standing_angle_threshold = 15  # Ngưỡng góc cho tư thế đứng
        self.prev_motion_values = []
        self.motion_history_size = 5
        self.playback_speed = 1.0  # Tốc độ phát mặc định
        self.paused = False  # Trạng thái pause
        
        # Khởi tạo Telegram bot
        self.bot_token = '7879096838:AAFfWOun8NuV0vCNNgY_IAOq7M3LL0wHi84'  # Thay thế bằng token của bạn
        self.chat_id = '-1002528969364'      # Thay thế bằng chat ID của bạn
        self.bot = telegram.Bot(token=self.bot_token)
        
        # Biến theo dõi thời gian ngã
        self.falling_start_time = None
        self.falling_alert_sent = False
        self.FALL_ALERT_THRESHOLD = 10  # Thời gian ngã (giây) trước khi gửi cảnh báo

    def preprocess_frame(self, frame, bbox):
        try:
            # Cắt vùng người từ frame gốc
            x, y, w, h = bbox
            
            # Thêm padding để đảm bảo lấy đủ context
            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            
            person = frame[y:y+h, x:x+w]
            
            # Giữ tỷ lệ khung hình
            target_size = max(self.input_shape)
            ratio = target_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            person = cv2.resize(person, new_size)
            
            # Pad để đạt kích thước yêu cầu
            delta_w = target_size - new_size[0]
            delta_h = target_size - new_size[1]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            person = cv2.copyMakeBorder(person, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=[0,0,0])
            
            # Resize về kích thước final
            person = cv2.resize(person, self.input_shape)
            
            # Normalize
            person = person.astype('float32') / 255.0
            person = np.expand_dims(person, axis=0)
            
            return person
        except Exception as e:
            print(f"Error in preprocess_frame: {e}")
            return None

    def detect_person(self, frame):
        # Chuyển sang RGB để xử lý với MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Lấy tọa độ các landmark và normalize về [0,1]
            landmarks = []
            h, w, _ = frame.shape
            for landmark in results.pose_landmarks.landmark:
                x, y = landmark.x, landmark.y  # Đã là số thập phân [0,1]
                x_pixel, y_pixel = int(x * w), int(y * h)  # Chuyển về pixel
                landmarks.append((x_pixel, y_pixel))
            
            # Tính bounding box
            x_coords = [x for x, y in landmarks]
            y_coords = [y for x, y in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Normalize bbox coordinates về [0,1]
            x_min = float(x_min) / w
            y_min = float(y_min) / h
            width = float(x_max - x_min) / w
            height = float(y_max - y_min) / h
            
            # Convert back to pixel coordinates for display
            x_min_pixel = int(x_min * w)
            y_min_pixel = int(y_min * h)
            width_pixel = int(width * w)
            height_pixel = int(height * h)
            
            # Vẽ skeleton
            self.mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            return (x_min_pixel, y_min_pixel, width_pixel, height_pixel)
        
        return None
    

    async def send_telegram_alert(self, frame):
        try:
            if not self.falling_alert_sent:
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
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=message
                    )
                self.falling_alert_sent = True
                print("Đã gửi cảnh báo qua Telegram!")
        except Exception as e:
            print(f"Lỗi khi gửi cảnh báo Telegram: {str(e)}")

    def predict_action(self, frame, bbox, pose_results, landmarks=None):
        try:
            processed_frame = self.preprocess_frame(frame, bbox)
            if processed_frame is None:
                return "No detection", 0.0
            
            prediction = self.model.predict(processed_frame, verbose=0)
            action_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][action_idx])
            predicted_action = self.actions[action_idx]

            if pose_results and pose_results.pose_landmarks:
                # Lấy các điểm mốc quan trọng
                left_hip = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_ankle = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

                # Tính góc thân người
                body_angle = abs(np.degrees(np.arctan2(
                    right_shoulder.y - right_hip.y,
                    right_shoulder.x - right_hip.x
                )))

                # Tính chiều cao và các tỷ lệ
                hip_y = (left_hip.y + right_hip.y) / 2
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                ankle_y = (left_ankle.y + right_ankle.y) / 2
                knee_y = (left_knee.y + right_knee.y) / 2
                
                height_ratio = abs(hip_y - shoulder_y) / abs(ankle_y - shoulder_y)
                knee_hip_ratio = abs(knee_y - hip_y) / abs(ankle_y - hip_y)
                
                # Tính toán mức độ chuyển động
                current_motion = self.calculate_motion(landmarks, frame.shape)
                
                # Kiểm tra các điều kiện cho từng hành động
                
                # 1. Kiểm tra điều kiện nằm/ngã
                is_lying = (
                    (body_angle > 45 and body_angle < 135) or  # Thân người nằm ngang (nới lỏng góc)
                    (height_ratio < 0.25) or  # Chiều cao rất thấp
                    (abs(hip_y - knee_y) < 0.08)  # Chân duỗi thẳng khi nằm
                )

                # 2. Kiểm tra điều kiện đứng
                is_standing = (
                    (body_angle > 75 and body_angle < 105) and  # Thân người thẳng đứng
                    (height_ratio > 0.35) and  # Chiều cao đủ cao
                    abs(left_hip.y - right_hip.y) < 0.08  # Hông cân bằng
                )

                # 3. Kiểm tra điều kiện ngồi
                is_sitting = (
                    (knee_hip_ratio < 0.5) and  # Đầu gối gần hông
                    (height_ratio > 0.2 and height_ratio < 0.4) and  # Chiều cao trung bình
                    (body_angle > 60 and body_angle < 120)  # Thân hơi nghiêng
                )

                # 4. Kiểm tra điều kiện chạy/đi bộ
                is_moving = current_motion > self.motion_threshold
                legs_moving = abs(left_knee.y - right_knee.y) > 0.1
                is_running = is_moving and legs_moving and (current_motion > self.motion_threshold * 2)

                # Quyết định hành động cuối cùng dựa trên các điều kiện và confidence
                if is_lying and current_motion < self.motion_threshold * 0.7:
                    predicted_action = 'falling'
                    confidence = max(0.9, confidence)
                    
                    # Xử lý cảnh báo ngã
                    current_time = time.time()
                    if self.falling_start_time is None:
                        self.falling_start_time = current_time
                    if not self.falling_alert_sent and (current_time - self.falling_start_time) >= self.FALL_ALERT_THRESHOLD:
                        asyncio.run(self.send_telegram_alert(frame))
                
                elif is_sitting:
                    predicted_action = 'sitting'
                    confidence = max(0.85, confidence)
                    self.falling_start_time = None
                    self.falling_alert_sent = False
                
                elif is_standing and current_motion < self.motion_threshold:
                    predicted_action = 'standing'
                    confidence = max(0.85, confidence)
                    self.falling_start_time = None
                    self.falling_alert_sent = False
                
                elif is_running:
                    predicted_action = 'running'
                    confidence = max(0.85, confidence)
                    self.falling_start_time = None
                    self.falling_alert_sent = False
                
                elif is_moving and is_standing:
                    predicted_action = 'walking'
                    confidence = max(0.85, confidence)
                    self.falling_start_time = None
                    self.falling_alert_sent = False
                
                # Nếu không thỏa mãn điều kiện nào, giữ nguyên kết quả từ model
                
            return predicted_action, confidence

        except Exception as e:
            print(f"Lỗi trong predict_action: {str(e)}")
            return "Error", 0.0

    def get_color(self, confidence):
        if confidence < 0.4:  # Tăng ngưỡng đỏ
            return (0, 0, 255)  # Đỏ
        elif confidence < 0.6:  # Tăng ngưỡng vàng
            return (0, 255, 255)  # Vàng
        else:
            return (0, 255, 0)  # Xanh

    def smooth_prediction(self, action, confidence):
        if confidence < self.confidence_threshold:
            return self.prev_action if self.prev_action else (action, confidence)
        
        if self.prev_action:
            prev_action, prev_conf = self.prev_action
            if prev_action == action:
                # Làm mượt confidence
                smoothed_conf = prev_conf * (1 - self.smooth_factor) + confidence * self.smooth_factor
                self.prev_action = (action, smoothed_conf)
                return self.prev_action
        
        self.prev_action = (action, confidence)
        return action, confidence

    def calculate_motion(self, current_landmarks, frame_shape):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return 1.0

        if current_landmarks is None:
            return 1.0

        total_motion = 0
        h, w = frame_shape[:2]
        
        # Tính toán chuyển động dựa trên các điểm landmark chính
        key_points = [0, 11, 12, 23, 24]  # Đầu, vai, hông
        for idx in key_points:
            if idx < len(current_landmarks) and idx < len(self.prev_landmarks):
                curr_x, curr_y = current_landmarks[idx]
                prev_x, prev_y = self.prev_landmarks[idx]
                
                # Chuẩn hóa khoảng cách theo kích thước frame
                motion = np.sqrt(((curr_x - prev_x)/w)**2 + ((curr_y - prev_y)/h)**2)
                total_motion += motion

        self.prev_landmarks = current_landmarks
        return total_motion / len(key_points)

    def calculate_body_angles(self, landmarks):
        if not landmarks or len(landmarks) < 33:  # MediaPipe có 33 landmarks
            return None
        
        try:
            # Lấy các điểm quan trọng
            hip_left = landmarks[23]  # POSE_LANDMARKS.LEFT_HIP
            hip_right = landmarks[24]  # POSE_LANDMARKS.RIGHT_HIP
            shoulder_left = landmarks[11]  # POSE_LANDMARKS.LEFT_SHOULDER
            shoulder_right = landmarks[12]  # POSE_LANDMARKS.RIGHT_SHOULDER
            knee_left = landmarks[25]  # POSE_LANDMARKS.LEFT_KNEE
            knee_right = landmarks[26]  # POSE_LANDMARKS.RIGHT_KNEE
            
            # Tính góc thân trên (giữa vai và hông)
            upper_angle = np.arctan2(
                (shoulder_left[1] + shoulder_right[1])/2 - (hip_left[1] + hip_right[1])/2,
                (shoulder_left[0] + shoulder_right[0])/2 - (hip_left[0] + hip_right[0])/2
            )
            upper_angle = np.degrees(upper_angle)
            
            # Tính góc chân (giữa hông và đầu gối)
            leg_angle = np.arctan2(
                (knee_left[1] + knee_right[1])/2 - (hip_left[1] + hip_right[1])/2,
                (knee_left[0] + knee_right[0])/2 - (hip_left[0] + hip_right[0])/2
            )
            leg_angle = np.degrees(leg_angle)
            
            return upper_angle, leg_angle
        except Exception as e:
            print(f"Error calculating angles: {e}")
            return None

    def is_vertical_motion(self, current_landmarks, prev_landmarks):
        if not current_landmarks or not prev_landmarks:
            return False
        
        # Lấy điểm hông (hip center) để theo dõi chuyển động theo chiều dọc
        hip_idx = 23  # LEFT_HIP
        if hip_idx < len(current_landmarks) and hip_idx < len(prev_landmarks):
            curr_y = current_landmarks[hip_idx][1]
            prev_y = prev_landmarks[hip_idx][1]
            
            # Tính chuyển động theo chiều dọc
            vertical_motion = abs(curr_y - prev_y)
            return vertical_motion > 20  # Ngưỡng chuyển động dọc
        
        return False

    def get_stable_action(self, action, confidence, frame_shape, landmarks):
        current_time = time.time()
        
        # Thêm vào buffer
        self.action_buffer.append((action, confidence, current_time))
        if len(self.action_buffer) > self.buffer_size:
            self.action_buffer.pop(0)
        
        # Nếu chưa đủ thời gian tối thiểu, giữ hành động cũ
        if current_time - self.last_action_time < self.min_action_duration:
            return self.prev_action if self.prev_action else (action, confidence)
        
        motion = self.calculate_motion(landmarks, frame_shape)
        angles = self.calculate_body_angles(landmarks)
        
        # Thêm action mới vào history
        self.action_history.append((action, confidence, motion))
        if len(self.action_history) > self.history_size:
            self.action_history.pop(0)
        
        # Đếm số lần xuất hiện của mỗi action trong buffer
        action_counts = {}
        action_confidences = {}
        for a, c, _ in self.action_buffer:
            if a not in action_counts:
                action_counts[a] = 0
                action_confidences[a] = []
            action_counts[a] += 1
            action_confidences[a].append(c)
        
        # Tìm hành động phổ biến nhất trong buffer
        if action_counts:
            max_count = max(action_counts.values())
            most_common_actions = [a for a, c in action_counts.items() if c == max_count]
            
            if len(most_common_actions) == 1:
                stable_action = most_common_actions[0]
                avg_confidence = np.mean(action_confidences[stable_action])
            else:
                # Nếu có nhiều hành động cùng số lần xuất hiện, chọn cái có confidence cao nhất
                stable_action = max(most_common_actions, 
                                 key=lambda x: np.mean(action_confidences[x]))
                avg_confidence = np.mean(action_confidences[stable_action])
        else:
            stable_action = action
            avg_confidence = confidence
        
        # Cập nhật thời gian nếu hành động thay đổi
        if self.prev_action and stable_action != self.prev_action[0]:
            self.last_action_time = current_time
        
        return stable_action, avg_confidence

    def run(self, video_path=None):
        if video_path is None:
            rtsp_url = 'rtsp://admin:hoanghue123@172.16.64.251:554/onvif2'
            cap = cv2.VideoCapture(rtsp_url)
        else:
            cap = cv2.VideoCapture(video_path)
            
            # Lấy thông tin video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            print(f"Video info:")
            print(f"- Total frames: {total_frames}")
            print(f"- FPS: {fps:.2f}")
            print(f"- Duration: {duration:.2f} seconds")
        
        prev_time = 0
        frame_skip = 2  # Chỉ xử lý 1 frame trong mỗi 2 frame
        frame_count = 0
        
        print("⚡ Hệ thống phát hiện ngã đã khởi động!")
        print("📱 Đã kết nối với Telegram Bot")
        print("⏳ Đang theo dõi...")
        
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                # Đọc thêm frame để tăng tốc độ
                if self.playback_speed > 1.0:
                    skip_frames = int(self.playback_speed) - 1
                    for _ in range(skip_frames):
                        ret, _ = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                
                frame_count += 1
                
                # Tính FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Detect người
                bbox = self.detect_person(frame)
                
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Lấy landmarks cho tính toán chuyển động
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(frame_rgb)
                    landmarks = []
                    if results.pose_landmarks:
                        h, w = frame.shape[:2]
                        for landmark in results.pose_landmarks.landmark:
                            x_pixel, y_pixel = int(landmark.x * w), int(landmark.y * h)
                            landmarks.append((x_pixel, y_pixel))
                    
                    # Dự đoán hành động
                    action, confidence = self.predict_action(frame, bbox, results, landmarks)
                    action, confidence = self.smooth_prediction(action, confidence)
                    action, confidence = self.get_stable_action(action, confidence, frame.shape, landmarks)
                    
                    # Hiển thị kết quả
                    text = f"{action}: {confidence:.2f}"
                    color = self.get_color(confidence)
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                              color, 2)
                    
                    # Hiển thị motion value
                    motion = self.calculate_motion(landmarks, frame.shape)
                    motion_text = f"Motion: {motion:.3f}"
                    cv2.putText(frame, motion_text, (10, frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Hiển thị tất cả confidence scores
                    all_confidences = self.model.predict(self.preprocess_frame(frame, bbox), verbose=0)[0]
                    for i, (act, conf) in enumerate(zip(self.actions, all_confidences)):
                        text = f"{act}: {conf:.2f}"
                        cv2.putText(frame, text, (10, 30 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  self.get_color(conf), 1)
                
                # Hiển thị FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Hiển thị trạng thái ổn định
                stability = min(len(self.action_buffer) / self.buffer_size, 1.0)
                stability_text = f"Stability: {stability:.2f}"
                cv2.putText(frame, stability_text,
                            (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                            self.get_color(stability), 1)
                
                # Hiển thị tốc độ phát
                speed_text = f"Speed: {self.playback_speed}x"
                cv2.putText(frame, speed_text,
                           (10, frame.shape[0] - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow('Action Detection', frame)
            
            # Tính thời gian chờ an toàn hơn
            base_wait = 30  # Thời gian chờ cơ bản (ms)
            if self.playback_speed >= 1.0:
                wait_time = max(1, int(base_wait / self.playback_speed))
            else:
                wait_time = min(100, int(base_wait / self.playback_speed))
            
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):  # Thoát
                break
            elif key == ord(' '):  # Pause/Resume
                self.paused = not self.paused
            elif key == ord(']'):  # Tăng tốc
                self.playback_speed = min(8.0, self.playback_speed + 0.5)
            elif key == ord('['):  # Giảm tốc
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
            elif key == ord('r'):  # Reset về tốc độ bình thường
                self.playback_speed = 1.0
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ActionDetector()
    
    # Đường dẫn đến file video (thay đổi theo nhu cầu)
    #video_path = "2.mp4"  # Ví dụ: "test_video.mp4"
    
    # Chạy với video file
    #detector.run(video_path)
    
    # Hoặc chạy với webcam
    detector.run()