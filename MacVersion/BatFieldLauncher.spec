# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# ------------------------------------------------------------
# 0) Imports for TensorFlow/Keras bundling (BatNoBat only)
# ------------------------------------------------------------
from PyInstaller.utils.hooks import (
    collect_submodules, collect_dynamic_libs, copy_metadata
)

# ------------------------------------------------------------
# 1) Common data (shared across all apps)
#    We place the PNGs next to the executables in dist.
# ------------------------------------------------------------
common_datas = [
    ('apps/batcrossingguard.png', '.'),      # BatNoBat1 image
    ('apps/BatCuttingNewspaper.png', '.'),    # BatCompressor1 image
    ('apps/BatSherlockHolms.png', '.'),      # BatInspector1 image
    # If you actually need this shared image, keep it; otherwise remove
    ('BatLookingAtBatCalls.png', '.'),
]

# ------------------------------------------------------------
# 2) TensorFlow/Keras support for BatNoBat only
# ------------------------------------------------------------
KERAS_HIDDEN = [
    *collect_submodules('keras'),
    *collect_submodules('tensorflow'),
]
TF_BINARIES = [
    *collect_dynamic_libs('tensorflow'),
    *collect_dynamic_libs('grpc'),
    *collect_dynamic_libs('absl'),
]
PKG_META = [
    *copy_metadata('keras'),
    *copy_metadata('tensorflow'),
]

# Your model file lives in apps/
NBN_DATAS = [
    ('apps/BatNoBat_Beta.keras', '.'),       # copied next to EXEs in dist
]

# ------------------------------------------------------------
# 3) App builders
#    - make_app: for launcher, compressor, inspector (no TF)
#    - make_app_nb: for BatNoBat (with TF/Keras assets)
# ------------------------------------------------------------
def make_app(script_path, exe_name):
    a = Analysis(
        [script_path],
        pathex=['.', 'apps'],
        binaries=[],
        datas=common_datas[:],
        hiddenimports=[],
        hookspath=[],
        runtime_hooks=[],
        excludes=['torch','keras.src.backend.torch'],
        noarchive=False,
    )
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=exe_name,                 # flat name, no subdir
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=False,
        console=False,                 # set True temporarily to debug
    )
    return a, exe

def make_app_nb(script_path, exe_name):
    a = Analysis(
        [script_path],
        pathex=['.', 'apps'],
        binaries=TF_BINARIES,               # TF dylibs
        datas=common_datas[:] + NBN_DATAS + PKG_META,
        hiddenimports=KERAS_HIDDEN,         # keras/tf submodules
        hookspath=[],
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
    )
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=False,
        console=False,                      # flip to True only if you need console logs
    )
    return a, exe

# ------------------------------------------------------------
# 4) Define the four apps
# ------------------------------------------------------------
a_launch, exe_launch = make_app('BatFieldLauncher.py',      'BatFieldLauncher')
a_comp,   exe_comp   = make_app('apps/BatCompressor1.py',   'BatCompressor1')
a_insp,   exe_insp   = make_app('apps/BatInspector1.py',    'BatInspector1')
a_nbn,    exe_nbn    = make_app_nb('apps/BatNoBat1.py',     'BatNoBat1')

# ------------------------------------------------------------
# 5) Single COLLECT for everything (NO MERGE)
#    Feed each app's binaries/zipfiles/datas directly.
# ------------------------------------------------------------
coll = COLLECT(
    exe_launch, exe_comp, exe_insp, exe_nbn,

    a_launch.binaries, a_launch.zipfiles, a_launch.datas,
    a_comp.binaries,   a_comp.zipfiles,   a_comp.datas,
    a_insp.binaries,   a_insp.zipfiles,   a_insp.datas,
    a_nbn.binaries,    a_nbn.zipfiles,    a_nbn.datas,

    strip=True,
    upx=False,
    upx_exclude=[],
    name='BatFieldTools'   # resulting dist folder name
)
