<h1 align="center">HỆ THỐNG NHẬN DẠNG HÀNH ĐỘNG CON NGƯỜI BẰNG CAMERA </h1>
<div align="center">

<p align="center">
  <img src="anh\logodnu.webp" alt="DaiNam University Logo" width="200"/>
  <img src="anh\mobile.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>


Dự án này phát triển hệ thống Nhận dạng và phân loại các hành động của con người (đi bộ, chạy, nhảy, ngã, đứng yên, ngồi) từ dữ liệu video. 
Ứng dụng trong nhiều lĩnh vực như giám sát an ninh, chăm sóc sức khỏe, và phân tích hành vi trong môi trường thông minh.

![image](https://github.com/user-attachments/assets/89f30396-29cc-4a91-85ef-077e71d83fcc)


## 📋 Tổng Quan

Hệ thống sử dụng mô hình YOLOv8 phiên bản small để nhận diện hành động con người qua camera, kết hợp với giao diện web để hiển thị kết quả và quản lý.
Mô hình hệ thống
![image](https://github.com/user-attachments/assets/713df677-ff07-432d-aca1-3b537edfc9f0)


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
nhan-dien-hanh-dong-con-nguoi/
├── app/                     # Thư mục chính của ứng dụng
│   ├── app.py               # Xử lý yêu cầu kết nối với camera và models
│   ├── server.py            # Xử lý yêu cầu kết nối với camera và models
│   ├── models/              # Mô hình YOLOv12
│   └── templates/           # HTML templates
│       └── index.html       # Giao diện người dùng chính
├── train_models.py           # Models được train
│── requirements.txt     # Danh sách thư viện cần thiết
├── Poster_CNTT16-06_Aiot_N10.pptx           # Poster nghiên cứu
```

## 📊 Dữ Liệu

Dữ liệu huấn luyện là tập dữ liệu riêng được thu thập bởi nhóm phát triển và hiện không được công khai. Nếu cần dữ liệu cho mục đích nghiên cứu, vui lòng liên hệ với tác giả.

## 🛠️ Công Nghệ Sử Dụng

- **Deep Learning**: YOLOv12, PyTorch, CNN, LSTM
- **Backend**: Flask, OpenCV
- **Frontend**: HTML
- **Phân tích dữ liệu**: NumPy, Pandas, Matplotlib


## Tỷ lệ Train và Test
Dữ liệu của 10 đối tượng (tình nguyện viên) được chia ngẫu nhiên thu được 151 video với tổng 44416 bức ảnh:
 - falling: 6779 samples
 - jumping: 7353 samples
 - running: 7066 samples
 - sitting: 7891 samples
 - standing: 6668 samples
 - walking: 8659 samples

![image](https://github.com/user-attachments/assets/92ac53be-c71c-49e1-9713-80417c27b985)

## Agenda

### 1. Phân tích dữ liệu (EDA)

-  Một số phân tích trên tập dữ liệu::
-  Đầu tiên, tiến hành EDA trên tập dữ liệu do chuyên gia tạo ra. Chúng tôi sẽ tìm hiểu dữ liệu và sau đó xây dựng một số mô hình Machine Learning trên tập dữ liệu này.
-  Tổng số điểm dữ liệu và số lượng đặc trưng trong tập huấn luyện và tập kiểm tra::
   ```python
      import os
      
      train_dir = "data\split_dataset\images/train"
      val_dir = "data\split_dataset\images/val"
      test_dir = "data\split_dataset\images/test"
      
      num_train = sum([len(files) for _, _, files in os.walk(train_dir)])
      num_val = sum([len(files) for _, _, files in os.walk(val_dir)])
      num_test = sum([len(files) for _, _, files in os.walk(test_dir)])
      
      print(f"Training samples: {num_train}")
      print(f"Validation samples: {num_val}")
      print(f"Testing samples: {num_test}")

   ```
   ```
   Output:
   Training samples: 31089
   Validation samples: 6663
   Testing samples: 6664
   ```

-  Phân tích số lượng mẫu theo từng lớp hoạt động:
      ![image](https://github.com/user-attachments/assets/93fed1bd-d5bd-4751-9598-2461dded31ab)
      *  Dữ liệu được phân bố khá đều giữa các lớp hoạt động.
          


### 3. Deep Learning Models:
Hệ thống sử dụng kiến trúc kết hợp CNN + LSTM:
- CNN: Được sử dụng để trích xuất đặc trưng từ các khung hình video.
- LSTM: Sử dụng các đặc trưng trích xuất từ CNN để mô hình hóa chuỗi thời gian và nhận diện hành động.
- Pipeline:
   + Dữ liệu video được chuyển thành chuỗi ảnh.
   + CNN trích xuất đặc trưng từ từng ảnh.
   + LSTM xử lý chuỗi đặc trưng để xác định hành động.
   
###   4.	Kết quả
 - Độ chính xác của mô hình CNN + LSTM đạt được 98% trên tập kiểm tra.
   ![image](https://github.com/user-attachments/assets/d1d17721-6c8a-4901-9ac4-e26eb7829538)



## Cài đặt
Mã được viết bằng Python 3.7. Nếu bạn chưa cài đặt Python, bạn có thể tìm thấy nó [**tại đây**](https://www.python.org/downloads/ "Cài đặt Python 3
.7"). Nếu bạn đang sử dụng phiên bản Python thấp hơn, bạn có thể nâng cấp bằng gói pip, đảm bảo bạn có phiên bản pip mới nhất.


  *How To*
  
    * Install Required Libraries
    
      ```python
      pip3 install pandas
      pip3 install numpy
      pip3 install scikit-learn
      pip3 install matplotlib
      pip3 install keras
      ```

## Quick overview of the dataset

- Cáchành động thu được từ camera được thực hiện từ 10 tình nguyện viên (gọi là đối tượng) trong khi thực hiện 6 hoạt động sau.
1. Đi bộ
2. Chạy
3. Nhảy
4. Ngã
5. Ngồi
6. Đứng.
- Mỗi khung hình thu được 30fps

## 📝 Liên Hệ

email: hoangphuonghue20@gmail.com


