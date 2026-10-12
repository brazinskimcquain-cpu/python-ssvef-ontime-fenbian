# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MEG-GO SSVEP.py'],
    pathex=[],
    binaries=[],
    datas=[('model', 'model'), ('layout', 'layout')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torchvision', 'tensorboard', 'optree', 'sympy', 'functorch', 'torch._dynamo', 'torch._inductor', 'torch.testing', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MEG-GO SSVEP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
