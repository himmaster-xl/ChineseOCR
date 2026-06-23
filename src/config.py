"""配置系统 — 从 YAML 文件解析并结构化存储所有超参数。

使用方式:
    cfg = Config.from_yaml("configs/vit_small_hwdb.yaml")
    print(cfg.model.hidden_dim)  # 384
"""

from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    """模型架构超参数"""
    # 必填
    in_channels: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    num_classes: int
    dropout: float
    # 可选
    image_size: int = 112
    patch_size: int = 8
    type: str = "resnet"


@dataclass
class TrainConfig:
    """训练过程超参数"""
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    label_smoothing: float
    grad_accum_steps: int


@dataclass
class DataConfig:
    """数据路径与加载超参数"""
    hdf5_path: str
    num_workers: int


@dataclass
class Config:
    """总的配置容器，组合模型、训练、数据三个子配置。

    使用 YAML 文件实例化:
        cfg = Config.from_yaml("path/to/config.yaml")
    """
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 配置文件加载所有配置。

        Args:
            path: YAML 配置文件路径

        Returns:
            Config 实例，包含 model / train / data 三个子配置
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            data=DataConfig(**raw["data"]),
        )


if __name__ == "__main__":
    import tempfile, os
    yaml_content = """
    model:
      image_size: 112
      patch_size: 4
      in_channels: 1
      hidden_dim: 384
      num_layers: 12
      num_heads: 6
      mlp_ratio: 4
      num_classes: 3755
      dropout: 0.1
    train:
      batch_size: 128
      epochs: 100
      lr: 0.0003
      weight_decay: 0.05
      warmup_epochs: 10
      label_smoothing: 0.1
      grad_accum_steps: 1
    data:
      hdf5_path: "data/processed/test.h5"
      num_workers: 4
    """
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    tmp.write(yaml_content)
    tmp.close()

    cfg = Config.from_yaml(tmp.name)
    assert cfg.model.hidden_dim == 384
    assert cfg.train.lr == 0.0003
    assert cfg.data.hdf5_path == "data/processed/test.h5"
    print("All config tests passed!")
    os.unlink(tmp.name)
