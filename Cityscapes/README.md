# DeepLabV3_plus
DeepLabV3+ is a state-of-art deep learning model for semantic image segmentation.

## Environment Requirements

    tensorflow-gpu==2.3.1


## Preparation
### 1. Download Cityscapes datasets
1. You need to register a new account on ` https://www.cityscapes-dataset.com/login/`
2. Modify the username & password & packageID in `datasets/download.sh`
3. Run `sh datasets/download.sh`

### 2. Split the train/val/test datasets
1. Modify the data path in `preparation/Cityscapes_train_tets_split.py` 
2. Run `python3 preparation/Cityscapes_train_tets_split.py`
3. Generate the txt files storing the RGB image and semantic image paths in folder `train_test_inputs`

### 3. Receptive fleid analysis

    https://fomoro.com/research/article/receptive-field-calculator#3,1,1,VALID;2,2,1,VALID;3,1,1,VALID;2,2,1,VALID;3,1,1,VALID;3,1,1,VALID;2,2,1,VALID