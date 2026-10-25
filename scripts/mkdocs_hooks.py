import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_examples_doc import generate_all


def on_pre_build(config):
    """Regenerate the docs/examples/ pages from examples/**/*.ipynb."""
    generate_all()
