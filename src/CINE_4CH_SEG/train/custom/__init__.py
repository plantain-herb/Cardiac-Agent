import importlib
import os
from pathlib import Path
from typing import List


def import_directory(dir_name: str, base_dir: str, ignore_packages: List[str] = []):
    ignore_dict = {}
    for p in ignore_packages:
        p = p.strip()
        if "/" in p:
            name, remain = p.split("/", 1)
            if len(remain) > 0:  # support 'utils/'
                if name in ignore_dict:
                    ignore_dict[name].append(remain)
                else:
                    ignore_dict[name] = [remain]
            else:
                ignore_dict[p] = None
        else:
            ignore_dict[p] = None

    for name in os.listdir(os.path.join(base_dir, dir_name)):
        if name in ["__pycache__"] or name.endswith(".pyc"):
            continue

        if name in ignore_dict and ignore_dict[name] is None:
            continue

        package_name = os.path.join(dir_name, name)
        if os.path.isdir(os.path.join(base_dir, package_name)):
            import_directory(package_name, base_dir, ignore_dict.get(name, []))
        elif os.path.isfile(os.path.join(base_dir, package_name)) and package_name.endswith(".py"):
            if "register_module" in Path(os.path.join(base_dir, package_name)).read_text():
                importlib.import_module("." + package_name.replace("/", ".")[:-3], package="custom")


try:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import_directory("", os.path.dirname(__file__), ignore_packages=["__init__.py", "utils/"])
    sys.path.pop(0)
except Exception as e:
    import sys
    import traceback

    print("*************error in import: {}******************".format(e))
    traceback.print_exc()
    sys.exit(1)
