# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('C:/Users/ilank/AppData/Local/Programs/Python/Python312/Lib/site-packages/faster_whisper/assets', 'faster_whisper/assets'),
    ],
    hiddenimports=[
        'ctranslate2',
        'faster_whisper',
        'huggingface_hub',
        'tokenizers',
        'pydantic',
        'pydantic.deprecated.decorator',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt6', 'tkinter', '_tkinter',
        'matplotlib', 'pygame', 'notebook', 'nbformat',
        'IPython', 'jupyter', 'black', 'yapf',
        # Large packages not used by the app
        'pyarrow', 'scipy', 'onnxruntime',
        'babel', 'pandas', 'sphinx', 'lxml',
        'cryptography', 'rapidfuzz',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='IvritTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    runtime_tmpdir=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IvritTranscriber',
)
