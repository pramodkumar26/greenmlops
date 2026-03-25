 
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "carbon"))


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")