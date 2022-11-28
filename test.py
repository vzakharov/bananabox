# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = dict(
  artist = "The Beatles",
  genre = "Rock",
  lyrics = """I get by with a little help from my friends
  When I was down and out
  They gave me something to talk about"""
)

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())