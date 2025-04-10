import json
import requests
from openai import OpenAI

with open("./config/keys.json") as f:
    config = json.load(f)
openai_api_key = config["openai_api_key"]

client = OpenAI(api_key=openai_api_key)

response = client.images.generate(
    model="dall-e-3",
    prompt="Ein junger Mann, stylisch, hipster",
    size="1024x1024",
    quality="standard",
    n=1
)

url = response.data[0].url
url_response = requests.get(url)

