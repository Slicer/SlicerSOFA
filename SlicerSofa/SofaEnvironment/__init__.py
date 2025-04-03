import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    script_dir + '/../../Sofa/lib/python3/site-packages',
] + sys.path

os.environ['SOFA_ROOT'] = script_dir + '/../../Sofa'

# Sofa, by default, will capture the exception handling. This is a workaround
# to keep the exception handling in PythonSlicer
defaultExecHook = sys.excepthook

import Sofa
import SofaRuntime

sys.excepthook = defaultExecHook

__all__ = ["Sofa", "SofaRuntime"]
