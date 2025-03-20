# Nhận dạng hành động con người bằng Camera

Dự án này phát triển hệ thống Nhận dạng và phân loại các hành động của con người (đi bộ, chạy, nhảy, ngã, đứng yên, ngồi) từ dữ liệu video. 
Ứng dụng trong nhiều lĩnh vực như giám sát an ninh, chăm sóc sức khỏe, và phân tích hành vi trong môi trường thông minh.
![image](https://github.com/user-attachments/assets/89f30396-29cc-4a91-85ef-077e71d83fcc)


## 📋 Tổng Quan

Hệ thống sử dụng mô hình YOLOv8 phiên bản small để nhận diện hành động con người qua camera, kết hợp với giao diện web để hiển thị kết quả và quản lý.

## 🔍 Thành Phần Chính

### 📷 Camera Module
- Thu thập hình ảnh từ camera
- Truyền dữ liệu qua đường stream

### 🖥️ Flask API
- Xử lý các yêu cầu từ Web UI
- Xử lý hình ảnh/video từ camera
- Áp dụng mô hình YOLOv8 để phát hiện người
- Lưu trữ kết quả và phân tích dữ liệu

### 🌐 Web UI
- Hiển thị video trực tiếp từ camera
- Hiển thị kết quả phát hiện đối tượng
- Cung cấp giao diện quản lý và thống kê

## ⚙️ Hướng Dẫn Cài Đặt

1. Clone repository này về máy:
   ```bash
   git clone [https://github.com/huehoang-204/nhan-dien-hanh-dong-con-nguoi.git]
   cd nhan-dien-hanh-dong-con-nguoi
   ```

2. Cài đặt các thư viện Python cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

- Lưu ý nếu bị thông báo lỗi attn ( Đây là một lớp attention ). Thì hãy chọn mô hình phiên bản (v) thấp hơn.
3. Cấu hình camera:
   - Lấy địa chỉ IP của camera
   - Thêm đường stream từ IP camera (xem thêm trong file `app.py` để hiểu cách cấu hình)

4. Khởi động máy chủ:
   ```bash
   python run.py
   ```

5. Truy cập giao diện web tại địa chỉ server đã cấu hình

## 📁 Cấu Trúc Dự Án

```
student-detection-system/
├── app/                     # Thư mục chính của ứng dụng
│   ├── app.py               # Xử lý yêu cầu kết nối với camera và models
│   ├── server.py            # Xử lý yêu cầu kết nối với camera và models
│   ├── models/              # Mô hình YOLOv12
│   └── templates/           # HTML templates
│       └── index.html       # Giao diện người dùng chính
├── train_models.py                   # Script khởi động ứng dụng
└── requirements.txt         # Danh sách thư viện cần thiết
```

## 📊 Dữ Liệu

Dữ liệu huấn luyện là tập dữ liệu riêng được thu thập bởi nhóm phát triển và hiện không được công khai. Nếu cần dữ liệu cho mục đích nghiên cứu, vui lòng liên hệ với tác giả.

## 🛠️ Công Nghệ Sử Dụng

- **Deep Learning**: YOLOv12, PyTorch, CNN, LSTM
- **Backend**: Flask, OpenCV
- **Frontend**: HTML
- **Phân tích dữ liệu**: NumPy, Pandas, Matplotlib

## 📝 Liên Hệ

[Thông tin liên hệ của bạn]

## 📜 Giấy Phép

[Thông tin giấy phép]

---
