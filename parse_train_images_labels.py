import numpy as np
import struct
from PIL import Image
import os

print("Processing started")

data_file = r'.\MNIST_data\train-images-idx3-ubyte\train-images.idx3-ubyte'
# 文件大小是47040016B,但实际数据是47040000B(需要减去16字节的文件头)
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'
data_buf = open(data_file, 'rb').read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

label_file = r'.\MNIST_data\train-labels-idx1-ubyte\train-labels.idx1-ubyte'
# 文件大小是60008B,但实际数据是60000B(需要减去8字节的文件头)
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'
label_buf = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'mnist_train'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

print("Processing images...")

for ii in range(numLabels):
    if ii % 1000 == 0:
        print(f"Processed {ii}/{numLabels} images...")
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + 'mnist_train_' + str(ii) + '.png'
    img.save(file_name)

print("Processing completed")
