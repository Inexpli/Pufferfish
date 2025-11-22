# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['pufferfish.py'],
    pathex=[],
    binaries=[],
    datas=[('models/policy_network/CN2_BN2_RLROP.onnx', 'models/policy_network'), ('models/policy_network/move_mapping.json', 'models/policy_network'), ('tablebases/gaviota', 'tablebases/gaviota'), ('tablebases/polyglot', 'tablebases/polyglot')],
    hiddenimports=['onnxruntime'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pufferfish',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pufferfish',
)
