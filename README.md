# 大语言模型潜在文化倾向性研究平台

当前版本：`0.1.0`

## 下载

- **Windows x64 便携版（无需安装 Python）**：打开 [Releases](https://github.com/diexie1/LLMs-Potential-Cultural-Tendencies/releases/latest)，下载 `LLM-Cultural-Tendencies-Windows-x64-*.zip`。
- **源码**：在仓库页面选择 **Code → Download ZIP**，或下载[主分支源码 ZIP](https://github.com/diexie1/LLMs-Potential-Cultural-Tendencies/archive/refs/heads/main.zip)。

便携版由 GitHub Actions 在 Windows 上构建，解压后双击 `LLM-Cultural-Orientation.exe` 即可启动。整个压缩包需要保留在同一个文件夹中。第一次启动时在浏览器工作台填写所用模型服务的 API 密钥。

## 从源码运行

源码运行需要 Python 3.10 或更新版本，并且首次安装依赖时需要联网。

### Windows

1. 安装 [Python](https://www.python.org/downloads/)；安装时启用 Python Launcher（`py`）或将 Python 加入 PATH。
2. 双击 `setup_windows.bat`，等待依赖安装完成。
3. 双击 `launch.bat`。也可以在 PowerShell 运行：

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

### macOS / Linux

```bash
python3 --version
sh setup.sh
sh run_macos.sh
```

启动后浏览器会打开本机工作台（`http://127.0.0.1:8765`）。关闭启动窗口或按 `Ctrl+C` 可停止服务。

## 添加自己的量表

公开仓库和便携版**不包含研究量表或问卷文件**。将你有权使用的 `.xlsx` 量表放入程序旁的 `data` 文件夹，在工作台刷新量表列表即可。

表格格式说明见 [`data/README.md`](data/README.md)：工作簿使用 `Sheet1`（英文）和 `Sheet2`（中文）；每张表的 A 列依次放置标题、作答说明和题目。没有量表文件时，平台仍可启动，但不会显示可运行的量表。

## API 密钥与结果

- 在工作台的设置中填写服务商、API 密钥和模型。密钥保存在本机 `user_config.json`，不会提交到 Git。
- 批次结果保存在 `results`，运行日志保存在 `logs`；这两个目录也不会提交到 Git。
- 只把你有权限公开的源码和材料放入公开仓库；不要提交 API 密钥、受限问卷或个人数据。

## Windows 版本构建

推送形如 `v0.1.0` 的 Git 标签后，GitHub Actions 会使用 Python 3.13 构建 Windows x64 单文件程序，并自动在该标签下创建 Release。Release 包只含平台程序和使用说明，不含量表。

## 署名

- 项目负责人：南京师范大学心理学院 谢恩慧
- 平台制作人：南京师范大学心理学院 李佳斌
