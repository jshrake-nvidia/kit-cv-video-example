# TODO: Work around OM-108110
# by explicitly adding the python3.dll directory to the DLL search path list.
# cv2.dll fails to load because it can't load the python3.dll dependency
try:
    import os
    import pathlib
    import sys

    # The python3.dll lives in the python directory adjacent to the kit executable
    # Get the path to the current kit process
    exe_path = sys.executable
    exe_dir = pathlib.Path(exe_path).parent
    python_dir = exe_dir / "python"
    print(f"Adding {python_dir} to DLL search path list")
    os.add_dll_directory(python_dir)
except Exception as e:
    print(f"Error adding python directory to DLL search path list {e}")

from .extension import *
