# In this file, we define download_model
# It runs during container build time to get model weights built into the container

#@title Download model weights
# (We'll use Colab syntax in case we're testing in Colab)

import os
import subprocess
import sys
import time


import app

if __name__ == "__main__":
  print('Downloading model...')
  app.init()
  # (The init function will download the model weights if they're not already downloaded. It will also take some time to actually load the model into memory, but that's okay because we're doing this during container build time)