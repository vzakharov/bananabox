# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
import subprocess
import time


def download_model():

  # Download the following models from https://openaipublic.azureedge.net/jukebox/models/ :
  # - 5b/vqvae.pth.tar
  # - 5b/prior_level_0.pth.tar
  # - 5b/prior_level_1.pth.tar
  # - 5b_lyrics/prior_level_2.pth.tar
  # To ~/.cache/jukebox/models

  for path in ['5b/vqvae.pth.tar', '5b/prior_level_0.pth.tar', '5b/prior_level_1.pth.tar', '5b_lyrics/prior_level_2.pth.tar']:
    remote_path = 'https://openaipublic.azureedge.net/jukebox/models/' + path
    local_path = os.path.expanduser('~/.cache/jukebox/models/' + path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
      print(f'Downloading {remote_path} to {local_path}')
      start = time.time()
      subprocess.run(['curl', '-L', remote_path, '-o', local_path])
      print(f'Downloaded to {local_path} in {time.time() - start} seconds')
    else:
      print(f'File {local_path} already exists')

print('Downloading model...')

if __name__ == "__main__":
  download_model()