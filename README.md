---
title: Segmentation for Virtual Try-On
emoji: 🎨
colorFrom: violet
colorTo: cyan
sdk: gradio
sdk_version: 4.0
app_file: app.py
pinned: false
---

# Segmentation for Virtual Try-On

Ứng dụng web phân đoạn người (Human Parsing) sử dụng YOLO segmentation model, được thiết kế cho ứng dụng Virtual Try-On. Ứng dụng được xây dựng với Gradio và có thể chạy trên Hugging Face Spaces.

## ✨ Tính năng

- 🎯 **Phân đoạn người tự động**: Phát hiện và phân đoạn các phần cơ thể (upperbody, lowerbody, wholebody)
- 🎨 **Giao diện trực quan**: UI hiện đại với overlay màu sắc và interactive tags
- 🔧 **Hỗ trợ nhiều model**: Tương thích với YOLO models (.pt, .pth, .onnx)
- 📊 **Hiển thị chi tiết**: Thông tin về confidence score, area ratio, và bounding boxes
- 🚀 **Deploy dễ dàng**: Sẵn sàng deploy trên Hugging Face Spaces hoặc Docker

## 🛠️ Cài đặt

### Yêu cầu

- Python 3.10+
- CUDA (khuyến nghị cho GPU)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài đặt model

Đặt file model YOLO (`human_parsing_11l.pt`) vào thư mục gốc hoặc `./models/human_parsing/`

## 🚀 Sử dụng

### Chạy local

```bash
python app.py
```

Ứng dụng sẽ chạy tại `http://localhost:7860`

### Sử dụng với Docker

```bash
docker build -t segmentation-vto .
docker run -p 7860:8080 segmentation-vto
```

### Deploy lên Hugging Face Spaces

1. Tạo một Space mới trên [Hugging Face](https://huggingface.co/spaces)
2. Push code lên Space:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

**Lưu ý**: Bạn cần sử dụng [Hugging Face Access Token](https://huggingface.co/settings/tokens) thay vì password để push code.

## 📁 Cấu trúc dự án

```
segmentation-vto/
├── app.py                 # Gradio application
├── human_parsing.py       # Human parsing model wrapper
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md            # Documentation
```

## 🎮 Hướng dẫn sử dụng

1. **Upload Model**: Chọn file model YOLO (.pt, .pth, hoặc .onnx)
2. **Upload Ảnh**: Chọn ảnh cần phân đoạn
3. **Segment**: Click nút "Segment Now" để bắt đầu
4. **Xem kết quả**: Kết quả sẽ hiển thị với overlay màu sắc và các tag tương tác

### Tham số mặc định

- **IoU Threshold**: 0.7
- **Confidence Threshold**: 0.3

## 🔧 Cấu hình

Bạn có thể điều chỉnh các tham số trong `app.py`:

```python
DEFAULT_IOU = 0.7      # IoU threshold
DEFAULT_CONF = 0.3     # Confidence threshold
SMALL_SEGMENT_RATIO = 0.02  # Tỷ lệ để đánh dấu segment nhỏ
```

## 📝 Model

Ứng dụng sử dụng YOLO segmentation model để phát hiện và phân đoạn các phần cơ thể:
- **upperbody**: Phần thân trên
- **lowerbody**: Phần thân dưới  
- **wholebody**: Toàn bộ cơ thể

Model được load tự động từ:
1. `human_parsing_11l.pt` (thư mục gốc)
2. `./models/human_parsing/human_parsing_11l.pt`

## 🌐 Hugging Face Space

Ứng dụng được deploy tại: [https://huggingface.co/spaces/vdt040499/segmentation-vto](https://huggingface.co/spaces/vdt040499/segmentation-vto)

## 📄 License

Dự án này được phát triển cho mục đích nghiên cứu và ứng dụng Virtual Try-On.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📧 Liên hệ

- GitHub: [vdt040499](https://github.com/vdt040499)
- Hugging Face: [vdt040499](https://huggingface.co/vdt040499)

---

**Lưu ý**: Đảm bảo bạn có quyền sử dụng model YOLO và tuân thủ các điều khoản sử dụng của model.

