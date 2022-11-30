# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = dict(
  artist = "The Beatles",
  genres = "Rock",
  lyrics = """I get by with a little help from my friends
  When I was down and out
  They gave me something to talk about"""
)


res = requests.post('http://4cc1-34-124-241-217.ngrok.io/', json = model_inputs)

print(res.json())