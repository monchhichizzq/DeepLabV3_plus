# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 0:35
# @Author  : Zeqi@@
# @FileName: counting_objects.py
# @Software: PyCharm
import json
import cv2

train_txt_file = '../train_test_inputs/Cityscape_eigen_train_files.txt'
val_txt_file = '../train_test_inputs/Cityscape_eigen_val_files.txt'
# Read data from pre-saved txt files
with open(train_txt_file, 'r') as f:
    train_filenames = f.readlines()
with open(val_txt_file, 'r') as f:
    val_filenames = f.readlines()

key_list = []
labels_dir = {}
index2color_dir = {0:  [255, 255, 255]}
index2name_dir = {0:  'unknown'}

def semantic_info():
    for filename in val_filenames:
        # print('')
        label_name = filename.split(' ')[-1]
        rgb_label_path =  label_name.split('.')[0] + '.png'
        rgb_label = cv2.imread(rgb_label_path)
        # print('rgb', rgb_label.shape)

        labelid_path = rgb_label_path.replace("color", "labelIds")
        labelid = cv2.imread(labelid_path, cv2.IMREAD_GRAYSCALE)
        # print('labelid', labelid.shape)

        label_path = label_name.split('.')[0] + '.json'
        label_path = label_path.replace("color", "polygons")
        with open(label_path) as label_json:
            label = json.load(label_json)

        object_labels = label['objects']
        # print(object_labels)

        for obj_label in object_labels:
            # print(obj_label)
            _label = obj_label['label']
            label_coordinate = obj_label['polygon'][0]
            # if _label == 'building':
            #     label_color = rgb_label[label_coordinate[1], label_coordinate[0]]
            #     label_id = labelid[label_coordinate[1], label_coordinate[0]]
            #     print(_label, label_id, label_color)
            if _label not in key_list:

                key_list.append(_label)
                # print(label_coordinate)
                label_color = rgb_label[label_coordinate[1], label_coordinate[0]]
                label_id = labelid[label_coordinate[1], label_coordinate[0]]
                index2color_dir[label_id] = label_color
                index2name_dir[label_id] = _label
                # print(label_coordinate)

    for filename in train_filenames:
        # print('')
        label_name = filename.split(' ')[-1]
        rgb_label_path =  label_name.split('.')[0] + '.png'
        rgb_label = cv2.imread(rgb_label_path)
        # print('rgb', rgb_label.shape)

        labelid_path = rgb_label_path.replace("color", "labelIds")
        labelid = cv2.imread(labelid_path, cv2.IMREAD_GRAYSCALE)
        # print('labelid', labelid.shape)

        label_path = label_name.split('.')[0] + '.json'
        label_path = label_path.replace("color", "polygons")
        with open(label_path) as label_json:
            label = json.load(label_json)

        object_labels = label['objects']
        # print(object_labels)

        for obj_label in object_labels:
            # print(obj_label)
            _label = obj_label['label']
            label_coordinate = obj_label['polygon'][0]

            # if _label == 'building':
            #     label_color = rgb_label[label_coordinate[1], label_coordinate[0]]
            #     label_id = labelid[label_coordinate[1], label_coordinate[0]]
            #     print(_label, label_id, label_color)

            if _label not in key_list:

                key_list.append(_label)
                # print(label_coordinate)
                label_color = rgb_label[label_coordinate[1], label_coordinate[0]]
                label_id = labelid[label_coordinate[1], label_coordinate[0]]
                index2color_dir[label_id] = label_color
                index2name_dir[label_id] = _label
                # print(label_coordinate)

                # if _label == 'building':
                #     print(_label, label_id, label_color)

    return index2color_dir,  index2name_dir

if __name__ == '__main__':
    index2color_dir, index2name_dir = semantic_info()
    print(len(index2color_dir))
    print(len(index2name_dir))
    print(index2color_dir)
    print(index2name_dir)


    # if label not in key_list:
    #     labels_dir[l]
