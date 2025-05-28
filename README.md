# LLaVA-Det-Seg

本项目包含基于 LLaVA、GroundingDINO 和 SAM 的聊天交互功能。

项目地址 : https://github.com/liuxiang09/LLaVA-Det-Seg

## 环境设置

1.  **克隆仓库:**
    如果您还没有克隆本项目，请先克隆：

    ```bash
    git clone https://github.com/liuxiang09/LLaVA-Det-Seg.git
    cd LLaVA-Det-Seg
    ```

2.  **创建 Python 环境:**
    推荐使用 Conda 或 venv 创建一个独立的 Python 环境：

    ```bash
    conda create -n your_env_name python=3.9 -y
    conda activate your_env_name
    ```

    请将 `your_env_name` 替换为您希望的环境名称。

3.  **安装依赖:**
    根据代码中的导入，您需要安装以下主要库。请注意，这可能不是完整的列表，具体依赖请参考项目中的 `setup.py` 或其他安装指南（如果存在）：

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # 根据您的 CUDA 版本调整
    pip install opencv-python pillow requests numpy gradio
    ```

    此外，本项目依赖于 `llava`、`groundingdino` 和 `segment_anything` 目录下的代码。请确保这些组件已正确设置（例如，作为子模块克隆，或按照其各自的说明进行安装）。

4.  **下载模型检查点:**
    您需要下载以下预训练模型检查点，并将它们放置在项目的 `checkpoints/` 目录下。请注意，本项目代码中使用的检查点文件名如下，如果您下载的文件名不同，请根据实际情况调整或在代码中进行修改：

    - LLaVA 模型: `llava-LDS-v3`
    - GroundingDINO 模型: `groundingdino_swint_ogc.pth`
    - SAM 模型: `sam_vit_h_4b8939.pth`

    **如何下载:**
    请查找这些模型文件的下载链接。通常，您可以在以下位置找到它们：

    - 本项目（LLaVA-Det-Seg）的 GitHub 仓库 Release 页面或 README 文件中是否有提供。
    - 原始模型（LLaVA、GroundingDINO、SAM）的官方仓库或 Hugging Face 页面上提供的预训练模型。

    找到链接后，您可以使用 `wget` 或 `curl` 等工具下载。例如：

    ```bash
    # 示例：下载 GroundingDINO 模型 
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P checkpoints/
    # 示例：下载 SAM 模型 
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8989.pth -P checkpoints/
    ```
    
    请确保所有必需的模型文件都已成功下载并存放在 `checkpoints/` 目录中。

## 使用说明

### 使用 chat_liux.py (命令行界面)

运行 `chat_liux.py` 脚本进行命令行交互。脚本会提示您输入图片路径和文本 prompt。

```bash
python chat_liux.py
```

根据提示输入图片路径（例如 `.asset/basketball-2.png`）和您的问题。
输出包括 LLaVA 的文本回复，以及检测框和分割掩码结果（如果适用），这些结果将保存在 `outputs/boxes` 和 `outputs/masks` 目录下。

### 使用 app.py (Gradio Web 应用)

运行 `app.py` 脚本启动 Gradio web 应用。应用启动后，您可以在浏览器中访问提供的 URL。

```bash
python app.py
```

在 web 界面中，您可以上传图片并输入文本 prompt 进行交互。界面将显示 LLaVA 的文本回复以及检测和分割的可视化结果。

请确保在运行这两个脚本之前，您已经完成了环境设置和模型检查点下载的步骤。
