"""
Hath_Yog_RAG Test Package
=========================
Contains automated unit and regression test suites.
"""

import os
import sys

# Ensure src directory and project root are in sys.path for test discovery
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_dir = os.path.join(_root_dir, "src")

for path in [_src_dir, _root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)
