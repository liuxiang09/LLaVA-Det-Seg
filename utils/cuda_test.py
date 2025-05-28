import torch
import os
from pathlib import Path

def check_cudnn():
    print("=== cuDNN 环境检查 ===")
    
    # 检查环境变量
    print("\n1. 环境变量:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '未设置')}")
    
    # 检查 PyTorch 信息
    print("\n2. PyTorch 信息:")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 是否可用: {torch.backends.cudnn.is_available()}")
    
    # 检查 cuDNN 文件
    print("\n3. cuDNN 文件位置:")
    search_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib'),
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"\n检查路径: {path}")
            cudnn_files = list(Path(path).glob('*cudnn*'))
            for f in cudnn_files:
                print(f"  发现: {f}")
                if os.path.islink(f):
                    real_path = os.path.realpath(f)
                    print(f"  链接指向: {real_path}")

if __name__ == "__main__":
    check_cudnn()