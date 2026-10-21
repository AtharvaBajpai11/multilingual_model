import traceback
import sys

from test_model import TestIsoFlowS2ST

test = TestIsoFlowS2ST()
test.setUp()

methods = [
    "test_01_s2st_encoder",
    "test_02_diflow_decoder",
    "test_03_duration_predictor",
    "test_04_isosync_bridge",
    "test_05_parameter_freezing",
    "test_06_gradient_flow",
    "test_07_full_pipeline_translation",
]

for m in methods:
    print(f"Running {m}...")
    try:
        getattr(test, m)()
    except Exception as e:
        print(f"ERROR in {m}: {e}")
        traceback.print_exc(file=sys.stdout)
