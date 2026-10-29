#!/usr/bin/env python3
"""Run weighted feature fusion, including matched-sample baselines.

Works in a terminal or VS Code Interactive. Outputs go to Weighted_Ridge.
"""

import sys
from Ridge_DWI_fMRI_fusion import main


if __name__ == "__main__":
    arguments = [] if "ipykernel" in sys.modules else sys.argv[1:]
    main(["--weighted", *arguments])
