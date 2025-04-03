import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(script_dir + '/../../python3'): #Install-tree
    os.environ['SOFA_ROOT'] = script_dir + '/../..'
    sys.path = [script_dir + '/../../python3/site-packages'] + sys.path
else:                                         #Build-tree
    os.environ['SOFA_ROOT'] = script_dir + '/../../../../../Sofa-build'
    sys.path = [script_dir + '/../../../../../Sofa-build/lib/python3/site-packages'] + sys.path

# Sofa, by default, will capture the exception handling. This is a workaround
# to keep the exception handling in PythonSlicer
defaultExecHook = sys.excepthook

import Sofa
import SofaRuntime

sys.excepthook = defaultExecHook

__all__ = ["Sofa", "SofaRuntime"]
