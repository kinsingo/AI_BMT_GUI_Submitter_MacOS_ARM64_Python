# # Do not touch this GUI_Mananger.py file.
import os, sys
import platform
from pathlib import Path
from collections.abc import Sequence

# macOS specific setup
if platform.system() == "Darwin":
    current_dir = Path(__file__).resolve().parent
    
    # Set local Qt Framework path first (for distribution without system Qt)
    local_lib_path = current_dir / "lib"
    current_dyld_framework_path = os.environ.get("DYLD_FRAMEWORK_PATH", "")
    if str(local_lib_path) not in current_dyld_framework_path:
        os.environ["DYLD_FRAMEWORK_PATH"] = f"{local_lib_path}:{current_dyld_framework_path}"
    
    # Set library path for dylib files
    current_dyld_library_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    if str(local_lib_path) not in current_dyld_library_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{local_lib_path}:{current_dyld_library_path}"
    
    # Set Qt plugin path to local PlugIns
    local_plugins_path = current_dir / "PlugIns"
    if local_plugins_path.exists():
        os.environ["QT_PLUGIN_PATH"] = str(local_plugins_path)

sys.path.append(str(Path(__file__).resolve().parent / "bin"))
import ai_bmt_interface_python as bmt
current_dir = Path(__file__).resolve().parent

# Set QT_PLUGIN_PATH if not already set by macOS setup above
if "QT_PLUGIN_PATH" not in os.environ:
    os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()

def ExecuteGUI(global_interface):
    args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
    # 1) Single interface
    if isinstance(global_interface, bmt.AI_BMT_Interface):
        return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Single_Task(args, global_interface)

    # 2) Sequence (excluding strings)
    if isinstance(global_interface, Sequence) and not isinstance(global_interface, (str, bytes)):
        interfaces = list(global_interface)
        if len(interfaces) == 0:
            raise ValueError("ExecuteGUI: An empty interface list was provided.")
        if not all(isinstance(x, bmt.AI_BMT_Interface) for x in interfaces):
            raise TypeError("ExecuteGUI: All items in the list must be instances of bmt.AI_BMT_Interface.")
        if len(interfaces) == 1:
            return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Single_Task(args, interfaces[0])
        else:
            return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Multiple_Tasks(args, interfaces)
        
    # Type mismatch
    raise TypeError(
        "ExecuteGUI: Argument must be a bmt.AI_BMT_Interface or a sequence (list/tuple, etc.) of them."
    )
