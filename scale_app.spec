# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 规格：大语言模型潜在文化倾向性研究 Web 端。

打包后请将 data/ 复制到可执行文件同级目录。
"""

import sys
from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

a = Analysis(
    ['main.py'],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / 'app' / 'templates'), 'app/templates'),
        (str(root / 'app' / 'static'), 'app/static'),
    ],
    hiddenimports=['openpyxl', 'openai', 'httpx', 'requests', 'flask', 'jinja2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LLM-Cultural-Orientation',
    icon=str(root / 'app' / 'static' / 'app.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='LLM-Cultural-Orientation.app',
        icon=None,
        bundle_identifier='com.nnnu.llm-cultural-orientation',
    )
