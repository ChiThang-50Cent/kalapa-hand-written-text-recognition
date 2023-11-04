# kalapa-hand-written-text-recognition
# Vietnamese Location Text Handwritten Characters Recognition - Convolution Recurrent Neural Nets

![image](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/fab16cb2-575a-479d-9f5d-cb4a069bb9b4)

Repo này sử dụng mạng CRNN để nhận biết chữ cái viết tay trong tiếng Việt, được huấn luyện dựa trên bộ dataset của cuộc thi Kalapa. Bộ dữ liệu gồm hình ảnh các địa chỉ cụ thể được viết tay và kèm theo nhãn dán cụ thể.

## Ví dụ

![image](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/421d9871-8649-4c14-b931-23f4462e7585)

<b>Text: </b> *Thôn 1 Ea Ktur Cư Kuin Đắk Lắk*

# Dataset

Dataset gồm có 2700 ảnh dành cho việc training và validation, có 540 ảnh dành cho việc test (public test, không có text kèm theo)

## Một số ví dụ

![15](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/266a15c9-4a30-4e24-a621-a01b0ea28f41)
*Phường 41 Quận 2 TP Hồ Chí Minh*

![16](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/fb4106da-3008-4a59-aae7-55c24c7b5d7f)
*Ấp Đồng Tâm Long Trạch Cần Đước Long An*

![17](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/d32c7885-a8fe-4e9d-8cae-ac9995969562)
*An Bình Cao Lãnh Đồng Tháp*

# Preprocessing

## Phân đoạn

Ở đây, mình có sử dụng Kmean để phân đoạn hình ảnh, cụ thể là gom bức ảnh thành hai cụm màu trắng và đen, đại diện cho phông nền (trắng) và chữ viết (đen) sau đó lưu ảnh dưới dạng binay</br>
Ví dụ:

  ![image](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/584d1034-290a-446f-afd3-95a363e4d634)
  ![image](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/9fa727a2-0520-492b-ad85-4195e18ddf78)

   *Ảnh gốc và ảnh sau khi phân đoạn, chia thành hai cụm chính*
## Transform

Do tập training chỉ có khoảng 2200 ảnh, nên mình có sử dụng một số phương pháp tăng cường ảnh như:
-  Random Rotation
-  Random Erasor
-  Elastic transform

Ảnh được mặc định resize về kích thước 64x768

# Training & Test
## Training options

Có các thông số sau có thể chỉnh sửa:

* binary: sử dụng ảnh dạng binary hay không, là ảnh thu được sau khi dùng Kmean để phân đoạn
* n_channels: dùng ảnh dạng RGB (dim=3) hay Grayscale hoặc binary (dim=1) mặc định bằng 1
* Các tham số khác như n_epoch, lr, batch_size

## Train

```
# for binay image 
python -W ignore train.py --root={} --model_save_path={} --lr=0.001 --batch_size=64 --n_epoch=60

# for grayscale image 
python -W ignore train.py --root={} --model_save_path={} --binary=0 --img_path='images' --lr=0.001 --batch_size=64 --n_epoch=60

# for rgb image 
python -W ignore train.py --root={} --model_save_path={} --binary=0 --img_path='images' --lr=0.001 --batch_size=64 --n_epoch=60 --n_channels=3
```

Model được mặc định train trên CUDA. (Có thể chạy trên Kaggle hoặc Colab, mất khoảng hơn 2m cho một epoch)

## Predict test set
Chạy trên test set và lưu dưới dạng csv, sử dụng hai thuật toán decode là Best Path và Beam Search
Chỉnh sửa save model path và transform cho từng loại train ở phía trên.

```
python test.py
```

# Demo
Code có sử dụng Flask để deploy lên web. Người dùng có thể truy cập đến thư mục demo và chạy lệnh sau:

```
cd demo
flask --app app run
```

Chạy lệnh và ta thu về được đường dẫn đến web demo ``` Running on http://127.0.0.1:5000 ```
Web cho phép upload ảnh chứa nội dung cần OCR thu được kết quả. Model hoạt động tốt nhất với nội dung về địa chỉ.

Kết quả:

![image](https://github.com/ChiThang-50Cent/kalapa-hand-written-text-recognition/assets/62085284/daa22d3a-0111-497a-bd3f-fb5acc771cfb)

