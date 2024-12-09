from transformers import JukeboxModel, JukeboxTokenizer
import torch as t

model_id = '../jukebox-5b-lyrics' #@param ['openai/jukebox-1b-lyrics', 'openai/jukebox-5b-lyrics']
sample_rate = 44100
total_duration_in_seconds = 200
raw_to_tokens = 128
chunk_size = 32
max_batch_size = 16

model = JukeboxModel.from_pretrained(
  '../jukebox-5b-lyrics',
  device_map = {'' : 'mps'},
  torch_dtype = t.float16,
  # cache_dir = f"{cache_path}/jukebox/models",
  offload_folder = 'weights',
  resume_download = True,
  min_duration = 0
).to("mps").eval()
# breakpoint()

# print("Model loaded: ", model)
print(f"Device: {model.device}")

def seconds_to_tokens(sec, level = 2):

  global sample_rate, raw_to_tokens, chunk_size

  tokens = sec * sample_rate // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size

  # For levels 1 and 0, multiply by 4 and 16 respectively
  tokens *= 4 ** (2 - level)

  return int(tokens)


n_samples = 4
generation_length = seconds_to_tokens(1)
offset = 0
level = 0

model.total_length = seconds_to_tokens(total_duration_in_seconds)

model_inputs = dict(

  artist = "Metallica",
  genres = "Metal",
  lyrics = """Say your prayers little one
Don't forget my son
To include everyone
Tuck you in, warm within
Keep you free from sin
Till the sandman he comes
"""

)

metas = model_inputs

labels = JukeboxTokenizer.from_pretrained(model_id)(**metas)['input_ids'][level].repeat(n_samples, 1).to("mps")

print(f"Labels: {labels}")

zs = [ t.zeros(n_samples, 0, dtype=t.long, device="mps") for _ in range(3) ]

print(f"Zs: {zs}")

# sampling_kwargs = dict(
#   temp = 0.98,
#   chunk_size = chunk_size,
# ) 

# zs = model.sample_partial_window(
#   zs, labels, offset, sampling_kwargs, level = level, tokens_to_sample = generation_length, max_batch_size = max_batch_size
# )

# Let's conver to model._sample, the parameters are:
# def _sample(
#     self,
#     music_tokens,
#     labels,
#     sample_levels,
#     metas=None,
#     chunk_size=32,
#     sampling_temperature=0.98,
#     lower_batch_size=16,
#     max_batch_size=16,
#     sample_length_in_seconds=24,
#     compute_alignments=False,
#     sample_tokens=None,
#     offset=0,
#     save_results=True,
#     sample_length=None,
# ) 

zs = model._sample( zs, labels, [level], metas, chunk_size, sample_length_in_seconds=1 )

print(f"Generated {len(zs)} samples: {zs}")