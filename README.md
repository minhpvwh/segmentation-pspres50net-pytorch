1. python data_download.py
2. dataset.py
    - là nơi trả ra các đường dẫn từ train và val qua hàm make_datapath_list()
    - class DataTransform có nhiệm vụ resize, scale ảnh lớn hơn hoặc nhỏ hơn ảnh ban đầu, xoay bức ảnh bn góc, bn độ,
    đưa về dạng tensor chuẩn hóa dữ liệu của nó. Các class trong Compose tìm trên mạng.
3. utils/augmentation.py chứa các class transform được định nghĩa trong class Compose
    - Scale hàm zoom up/down để lấy các góc khác nhau của ảnh nhưng vẫn giữ nguyên kích thước ảnh gốc
    - RandomRotation xoay ảnh nghiêng
    - RandomMirror lật ảnh
    - Resize để resize ảnh
    - Normalize chuẩn hóa ảnh

    - MyDataset

4. tạo model
    - 1. FeatureMapConvolution
    - 2. ResidualBlockPSP
    - 3. PyramidPooling
    - 4. DecoderModule
    - 5. AuxLossModule


5. tạo script để train
    # model
    # loss
    # optimizer
    # scheduler
    # train_model

