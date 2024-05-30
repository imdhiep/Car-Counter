from ultralytics import YOLO  # Import mô hình YOLO từ thư viện ultralytics
import cv2  # Import thư viện OpenCV để xử lý ảnh
import numpy as np  # Import thư viện NumPy để làm việc với các mảng
from sort import *  # Import tất cả các hàm từ file sort

# Khởi tạo mô hình YOLO với trọng số đã được huấn luyện sẵn 'yolov8n.pt'
model = YOLO('yolov8n.pt')

# Định nghĩa các lớp đối tượng mà mô hình có thể nhận diện
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
           "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", 
           "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
           "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
           "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
           "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
           "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
           "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Đọc video từ đường dẫn được chỉ định
cap = cv2.VideoCapture('./assets/videos/cars.mp4')  # Thay đường dẫn tới file video của bạn

# Kiểm tra xem video có mở được hay không
if cap.isOpened():
    # Lấy chiều rộng và chiều cao của khung hình video
    width = cap.get(3)  # Tham số 3: Chiều rộng của khung hình
    height = cap.get(4)  # Tham số 4: Chiều cao của khung hình
    # print(f'width = {width}, height = {height}')
    # Khởi tạo đối tượng VideoWriter để ghi video kết quả
    # output = cv2.VideoWriter('./result/result.mp4', 
    #                          cv2.VideoWriter_fourcc(*'mp4v'),  # Định dạng mã hóa video
    #                          30,  # Số khung hình trên giây
    #                          (int(width), int(height)))  # Kích thước khung hình video

# Đọc ảnh mặt nạ từ đường dẫn được chỉ định
mask = cv2.imread('./assets/images/mask.jpg')  # Thay đường dẫn tới file ảnh mặt nạ của bạn

# Thay đổi kích thước của ảnh mặt nạ theo kích thước của khung hình video
mask = cv2.resize(mask, (int(width), int(height)))

# Khởi tạo bộ theo dõi đối tượng với các tham số max_age, min_hits, iou_threshold
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.4)
# max_age: Số khung hình tối đa để theo dõi một đối tượng nếu không phát hiện được mới
# min_hits: Số lần phát hiện liên tiếp tối thiểu để xác nhận một đối tượng
# iou_threshold: Ngưỡng IOU (Intersection Over Union) để xác định đối tượng trùng lặp

# Định nghĩa tọa độ của đường đếm đối tượng
limit = [380, 290, 700, 290]  # Điều chỉnh [x1, y1, x2, y2] theo video và mặt nạ của bạn

# Khởi tạo danh sách chứa các ID của các đối tượng đã được đếm và biến đếm tổng số đối tượng
total_count = []
count = 0

while True:
    flag, img = cap.read()  # Đọc từng khung hình từ video
    if not flag:
        break  # Thoát khỏi vòng lặp nếu không còn khung hình nào

    # Áp dụng mặt nạ lên khung hình
    mask_region = cv2.bitwise_and(mask, img)
    
    # Nhận diện các đối tượng trong khung hình đã áp dụng mặt nạ
    results = model(mask_region, stream=True)
    
    # Khởi tạo mảng rỗng để chứa các bounding boxes và độ tin cậy
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ của bounding box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = int(box.cls[0])  # Lấy lớp đối tượng
            # cv2.putText(img, classes[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            conf = box.conf[0]  # Lấy độ tin cậy của bounding box
            # Kiểm tra điều kiện để thêm bounding box vào danh sách phát hiện
            if conf > 0.4 and (classes[label] == 'bicycle' or classes[label]=='car' or classes[label]=='motorbike' \
                or classes[label]=='bus' or classes[label]=='truck'):
                # cv2.putText(img, f'{classes[label]}{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)           
                current_stat = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_stat))  # Thêm bounding box vào mảng phát hiện
    
    # Cập nhật các đối tượng theo dõi với các bounding boxes mới
    result_tracker = tracker.update(detections)
    cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 4)  # Vẽ đường đếm đối tượng
    
    for trk in result_tracker:
        x1, y1, x2, y2, id = map(int, trk)  # Lấy tọa độ và ID của đối tượng theo dõi
        cx = (x1+x2)//2  # Tính tọa độ trung tâm của bounding box
        cy = (y1+y2)//2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box quanh đối tượng
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)  # Vẽ hình tròn tại tọa độ trung tâm
        cv2.putText(img, f'id: {id}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)  # Hiển thị ID của đối tượng
        # Kiểm tra xem đối tượng có nằm trên đường đếm không
        if (limit[0]<= cx <= limit[2]) and (limit[1]-30 <= cy <= limit[3]+30):
            if id not in total_count:
                total_count.append(id)  # Thêm ID vào danh sách đã đếm
                count += 1  # Tăng biến đếm tổng số đối tượng
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 4)  # Đổi màu đường đếm khi có đối tượng đếm được
    
    cv2.putText(img, f'total count: {count}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)  # Hiển thị tổng số đối tượng đã đếm được           
    
    # output.write(img)
    cv2.imshow('result', img)  # Hiển thị khung hình với các đối tượng được nhận diện

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn phím 'q' để thoát
        break

cap.release()  # Giải phóng bộ nhớ của video
# output.release()
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ
