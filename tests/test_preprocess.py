"""预处理模块测试"""
import struct, tempfile, os, numpy as np, h5py, sys
sys.path.insert(0, 'src')

from data.preprocess import parse_gnt_file, create_hdf5


class TestGNTParser:
    """GNT 二进制格式解析测试"""

    def test_parse_single_sample(self):
        """解析一个最小 GNT 样本"""
        # 构造一个虚拟 GNT 样本
        label = "啊"  # GB2312 编码: 0xB0 0xA1
        tag_code = label.encode("gb2312-80")
        # 模拟一张 2x2 的灰度图
        bitmap = bytes([0, 255, 128, 64])  # 4 字节
        width, height = 2, 2

        # GNT 格式: [样本大小(4B)] + [标签(2B)] + [宽(2B)] + [高(2B)] + [位图数据]
        # sample_size = 标签(2) + 宽(2) + 高(2) + 位图(w*h) = 6 + 4 = 10
        sample_data = (
            struct.pack("<I", 2 + 2 + 2 + width * height)
            + tag_code
            + struct.pack("<HH", width, height)
            + bitmap
        )
        # GNT 文件格式: [样本数(4B)] + 样本1 + 样本2 + ...
        file_data = struct.pack("<I", 1) + sample_data

        tmp = tempfile.NamedTemporaryFile(suffix='.gnt', delete=False)
        tmp.write(file_data)
        tmp.close()

        images, labels = parse_gnt_file(tmp.name)
        os.unlink(tmp.name)

        assert len(images) == 1
        assert labels[0] == "啊"
        assert images[0].shape == (2, 2)

    def test_create_hdf5(self):
        """HDF5 创建和结构验证"""
        images = [
            np.random.randint(0, 255, (112, 112), dtype=np.uint8)
            for _ in range(10)
        ]
        label_map = {f"char_{i}".encode(): i for i in range(10)}
        class_indices = list(range(10))

        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()

        create_hdf5(images, label_map, class_indices, tmp.name,
                     split='train', resize_to=112)

        with h5py.File(tmp.name, 'r') as f:
            assert 'train/images' in f
            assert f['train/images'].shape == (10, 112, 112)
            assert len(f['train/labels']) == 10

        os.unlink(tmp.name)
