import os
import struct

import numpy as np
from PIL import Image


def prepare_test_data(
    data_file: str = r"./MNIST_data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
    label_file: str = r"./MNIST_data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
    output_root: str = "mnist_test",
) -> None:
    print("Processing started")

    # 7840016B总长度中包含16B头信息,有效图像数据是7840000B
    data_file_size = 7840016
    data_file_size = str(data_file_size - 16) + "B"
    data_buf = open(data_file, "rb").read()
    _, num_images, num_rows, num_columns = struct.unpack_from(">IIII", data_buf, 0)
    datas = struct.unpack_from(">" + data_file_size, data_buf, struct.calcsize(">IIII"))
    datas = np.array(datas).astype(np.uint8).reshape(num_images, 1, num_rows, num_columns)

    # 10008B总长度中包含8B头信息,有效标签数据是10000B
    label_file_size = 10008
    label_file_size = str(label_file_size - 8) + "B"
    label_buf = open(label_file, "rb").read()
    _, num_labels = struct.unpack_from(">II", label_buf, 0)
    labels = struct.unpack_from(">" + label_file_size, label_buf, struct.calcsize(">II"))
    labels = np.array(labels).astype(np.int64)

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    for i in range(10):
        class_dir = output_root + os.sep + str(i)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    print("Processing images...")

    for ii in range(num_labels):
        if ii % 1000 == 0:
            print(f"Processed {ii}/{num_labels} images...")
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = output_root + os.sep + str(label) + os.sep + "mnist_test_" + str(ii) + ".png"
        img.save(file_name)

    print("Processing completed")


if __name__ == "__main__":
    prepare_test_data()
