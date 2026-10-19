"""
模型设备工具模块：CUDA 不可用时自动回退 CPU，避免推理阶段才报错。
"""


def resolve_device(configured: str | None, default: str = "cpu") -> str:
    """
    解析推理设备：配置了 cuda 但本机 torch 不支持时自动回退 cpu。

    :param configured: 配置的设备名（cpu/cuda/cuda:0/None）
    :param default: 无配置时的默认设备
    :return: 实际可用设备名
    """
    device = (configured or default or "cpu").strip()
    if not device.lower().startswith("cuda"):
        return device
    try:
        import torch  # 延迟导入，CPU 环境无需安装 torch 也能用 API 提供方

        if not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device


def resolve_fp16(configured: bool, device: str) -> bool:
    """
    解析是否启用 fp16：CPU 设备强制关闭（CPU 版 torch 不支持半精度）。
    """
    if not configured:
        return False
    return not str(device).lower().startswith("cpu")
