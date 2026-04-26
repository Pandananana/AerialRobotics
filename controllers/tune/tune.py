import os
import runpy
import sys

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main"))
sys.path.insert(0, main_dir)
os.chdir(main_dir)
runpy.run_path(os.path.join(main_dir, "tune.py"), run_name="__main__")
