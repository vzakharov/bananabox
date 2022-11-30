# In this file, we define download_model
# It runs during container build time to get model weights built into the container

#@title Download model weights
# (We'll use Colab syntax in case we're testing in Colab)

import os
import subprocess
import sys
import time

testing_with_colab = 'google.colab' in sys.modules
# (Depending on this, we'll either download the model weights locally or to a mounted Google Drive)


cache_path = None

def download_model():

  global cache_path

  remote_base = 'https://openaipublic.azureedge.net/jukebox/models/'

  if testing_with_colab:

    cache_path = '/content/drive/My Drive/jukebox-webui/_data/'
    # Connect to your Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

  else:
    cache_path = '~/.cache/'

  # Expand the ~ if it's there
  if cache_path.startswith('~'):
    cache_path = os.path.expanduser(cache_path)

  # Download the following models from the remote_base to the cache_path
  # - 5b/vqvae.pth.tar
  # - 5b/prior_level_0.pth.tar
  # - 5b/prior_level_1.pth.tar
  # - 5b_lyrics/prior_level_2.pth.tar

  for path in ['5b/vqvae.pth.tar', '5b/prior_level_0.pth.tar', '5b/prior_level_1.pth.tar', '5b_lyrics/prior_level_2.pth.tar']:
    remote_path = remote_base + path
    local_path = cache_path + 'jukebox/models/' + path
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