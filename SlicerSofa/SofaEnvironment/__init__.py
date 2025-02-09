import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(script_dir + '/../../../../../Sofa-build'): # Build tree
    os.environ['SOFA_ROOT'] = script_dir + '/../../../../../Sofa-build'
else: # Install tree
    # Sofa does not allow much configurability of the install tree, therefore it is needed
    # to add extra python paths.
    sys.path = [
        script_dir + '/../../../../plugins/SofaPython3/lib/python3/site-packages',
        script_dir + '/../../../../plugins/STLIB/lib/python3/site-packages'
    ] + sys.path

    os.environ['SOFA_ROOT'] = script_dir + '/../../../../'

# Sofa, by default, will capture the exception handling. This is a workaround
# to keep the exception handling in PythonSlicer
defaultExecHook = sys.excepthook

import Sofa
import SofaRuntime

sys.excepthook = defaultExecHook

__all__ = ["Sofa", "SofaRuntime"]
