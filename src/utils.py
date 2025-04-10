import json
import glob
import pprint
from openai import OpenAI
from pydantic import BaseModel


def make_request(prompt_text, additional_information_list=None, image_paths=None):
    messages = [
        {"role": "system", "content": prompt_text},
    ]
    if additional_information_list is not None and len(additional_information_list) > 0:
        for additional_information in additional_information_list:
            messages.append({"role": "system", "content": additional_information})
    if image_paths is not None and len(image_paths) > 0:
        for image_path in image_paths:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Beachte dabei auch folgendes Bild:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                        },
                    },
                ],
            })
    with open("./src/config/keys.json") as f:
        config = json.load(f)
    openai_api_key = config["openai_api_key"]

    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return completion.choices[0].message.content

class InformationChunk(BaseModel):
    content: str
    source: str

class ResponseChunk(BaseModel):
    information_chunks: list[InformationChunk]

json_schema = {
  "name": "text_points",
  "schema": {
    "type": "object",
    "properties": {
      "points": {
        "type": "array",
        "description": "A list of key points extracted from the text.",
        "items": {
          "type": "object",
          "properties": {
            "content": {
              "type": "string",
              "description": "The actual text of the key point."
            },
            "importance": {
              "type": "string",
              "description": "The level of importance or relevance of the point."
            },
            "reference": {
              "type": "string",
              "description": "Reference to where this point can be found in the text."
            }
          },
          "required": [
            "content",
            "importance",
            "reference"
          ],
          "additionalProperties": False
        }
      }
    },
    "required": [
      "points"
    ],
    "additionalProperties": False
  },
  "strict": True
}

def make_request_structured(prompt_text, additional_information_dict=None, image_paths=None, json_schema=None):
    if additional_information_dict is None:
        additional_information_dict = {}
    messages = [
        {"role": "system", "content": prompt_text},
    ]
    if additional_information_dict is not None and len(additional_information_dict) > 0:
        messages.append({"role": "system", "content": "Beachte dabei auch folgende zusätzliche Informationen:"})
        for source, text in additional_information_dict.items():
            additional_information = f"Source: {source}\nContent: {text}"
            messages.append({"role": "system", "content": additional_information})
    if image_paths is not None and len(image_paths) > 0:
        for image_path in image_paths:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Beachte dabei auch folgendes Bild:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                        },
                    },
                ],
            })
    with open("./src/config/keys.json") as f:
        config = json.load(f)
    openai_api_key = config["openai_api_key"]

    client = OpenAI(api_key=openai_api_key)

   # print(json.dumps(messages, indent=2, ensure_ascii=False)) # for debug

    if json_schema is None:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages
        )
    else:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )
    return completion.choices[0].message.content

def load_prompt(filename_pattern):
    file_paths = glob.glob(f"./canned_prompts/{filename_pattern}.txt")
    if len(file_paths) == 0:
        return None
    with open(file_paths[0], 'r', encoding='utf8') as file:
        # Read the entire content of the file
        return file.read()

def load_schema(filename_pattern):
    file_paths = glob.glob(f"./json_schemas/{filename_pattern}.json")
    if len(file_paths) == 0:
        return None
    with open(file_paths[0], 'r', encoding='utf8') as file:
        # Read the entire content of the file
        return json.load(file)


