# Zalo AI 2021 5k Compliance 

## Team OverCoded 
- Hieu
- Hung 

## Data 

Sample submission: https://dl.challenge.zalo.ai/5k-compliance/sample_submission.zip

Pubic test: https://dl.challenge.zalo.ai/5k-compliance/public_test.zip

Train dataset: https://dl.challenge.zalo.ai/5k-compliance/train.zip

**Tải 3 link trên rồi để vào thư mục data dưới dạng : ./data/public_test/<giải nén> và ./data/train/<giải nén> và ./sample-submission/<giải nén>. File zip để chung thư mục hoặc xóa luôn cũng được.**


Data Train:

Inside the train.zip, you can find the following:
- images: folder stores image file
- train_meta.csv file: includes 5 columns:
    - image_id: id of image
    - fname: filename of image
    - mask: mask label
    - distancing: distance label
    - 5k: meet both conditions of mask and distancing
 
**Please note that there are some missing labels in the training dataset.**

Data Public Test:

Inside the public_test.zip, you can find the following:
- images: folder stores list image files (jpg)
- public_test_meta.csv file: includes 2 columns:
    - image_id: id of image
    - fname: filename of image




# temp
mask: >= 1 person  (f1) + missing labels 

yolov5 -> detect person -> 

đầu vào là hình người và đầu ra là có khẩu trang hay không 
yolov5 face mask detection model -> khẩu trang : yes/no -> prob: chưa giải quyết được cái đeo không đúng quy định

ensemble 

detect mặt người -> không tối ưu 
efficientnet -> có detect mask -> không tối ưu 

6 người -> 0 or 1 

team cv : face mask detection có model và data -> yolov5


distancing : 2m -> 

adjacent matrix 

[[1, 1],
 [1, 1]]
[[vertex],
 [vertex]]


detect person -> depth prediction + bboxes -> đại khái là tọa độ 3 chiều -> graphcnn classfication

fill missing labels -> 



datacomp -> chưa clean -> chưa cân bắng 

-> 