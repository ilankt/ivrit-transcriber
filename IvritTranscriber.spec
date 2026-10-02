# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
import sys

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=collect_data_files('faster_whisper', includes=['assets/*']) + [
        ('ICON.png', '.'),
        ('Binaries', 'Binaries'),
        ('Models', 'Models'),
    ],
    hiddenimports=[
        'ctranslate2',
        'faster_whisper',
        'huggingface_hub',
        'tokenizers',
        'pydantic',
        'pydantic.deprecated.decorator',
        'sounddevice',
        '_sounddevice_data',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt6', 'tkinter', '_tkinter',
        'matplotlib', 'pygame', 'notebook', 'nbformat',
        'IPython', 'jupyter', 'black', 'yapf',
        # Large packages not used by the app
        'pyarrow', 'scipy',
        'babel', 'pandas', 'sphinx', 'lxml',
        'cryptography', 'rapidfuzz',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Qt uses the ICU API supplied by Windows. A third-party icuuc.dll found on
# PATH (for example, Poppler's version) has incompatible, versioned exports.
# Let the Windows loader resolve its own ICU libraries instead of bundling them.
if sys.platform == 'win32':
    a.binaries = [
        entry for entry in a.binaries
        if entry[0].replace('\\', '/').rsplit('/', 1)[-1].lower()
        not in {'icuuc.dll', 'icuin.dll', 'icudt.dll'}
    ]

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
    icon='icon.ico',
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
