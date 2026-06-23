"""预处理模块测试"""
import struct, tempfile, os, numpy as np, h5py, sys
sys.path.insert(0, 'src')

from data.preprocess import parse_gnt_file, create_hdf5, tag_to_char


class TestGNTParser:
    """GNT 二进制格式解析测试"""

    def _make_gnt(self, samples):
        """构造 GNT 文件内容。每个 sample: (tag_value, width, height, bitmap_bytes)"""
        buf = bytearray()
        for tag_val, w, h, bmp in samples:
            sample_size = 4 + 2 + 2 + 2 + w * h  # 包含自身4B
            buf += struct.pack("<I", sample_size)
            buf += struct.pack("<H", tag_val)
            buf += struct.pack("<H", w)
            buf += struct.pack("<H", h)
            buf += bmp
        return bytes(buf)

    def test_parse_single_sample(self):
        """解析一个最小 GNT 样本"""
        bmp = bytes([0, 128, 255, 64])  # 2x2
        data = self._make_gnt([(33, 2, 2, bmp)])

        tmp = tempfile.NamedTemporaryFile(suffix='.gnt', delete=False)
        tmp.write(data)
        tmp.close()

        images, tags = parse_gnt_file(tmp.name)
        os.unlink(tmp.name)

        assert len(images) == 1
        assert tags[0] == 33
        assert images[0].shape == (2, 2)

    def test_parse_multiple_samples(self):
        """解析多个样本"""
        samples = [
            (0x21, 3, 5, bytes(15)),   # 3x5 = 15
            (0xA1A1, 4, 4, bytes(16)),  # 4x4 = 16
            (0xFEF7, 6, 2, bytes(12)),  # 6x2 = 12
        ]
        data = self._make_gnt(samples)

        tmp = tempfile.NamedTemporaryFile(suffix='.gnt', delete=False)
        tmp.write(data)
        tmp.close()

        images, tags = parse_gnt_file(tmp.name)
        os.unlink(tmp.name)

        assert len(images) == 3
        assert tags == [0x21, 0xA1A1, 0xFEF7]
        assert images[0].shape == (5, 3)   # (height, width)
        assert images[1].shape == (4, 4)
        assert images[2].shape == (2, 6)

    def test_create_hdf5(self):
        """HDF5 创建和结构验证"""
        images = [
            np.random.randint(0, 255, (112, 112), dtype=np.uint8)
            for _ in range(10)
        ]
        indices = list(range(10))

        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()

        create_hdf5(images, indices, tmp.name, split='train', resize_to=112)

        with h5py.File(tmp.name, 'r') as f:
            assert 'train/images' in f
            assert f['train/images'].shape == (10, 112, 112)
            assert len(f['train/labels']) == 10

        os.unlink(tmp.name)


class TestTagDecoding:
    """标签解码测试"""

    def test_ascii_tag(self):
        """ASCII 范围标签: 高字节为0x00"""
        assert tag_to_char(0x0021) == '!'   # 33 -> '!'
        assert tag_to_char(0x0041) == 'A'   # 65 -> 'A'
        assert tag_to_char(0x0030) == '0'   # 48 -> '0'

    def test_chinese_tag(self):
        """中文标签: GB2312 双字节"""
        # '的' = GB2312 B5 C4, uint16 LE = 0xC4B5
        assert tag_to_char(0xC4B5) == '的'
