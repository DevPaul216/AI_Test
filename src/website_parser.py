import json
import os
import requests

from openai import OpenAI
from bs4 import BeautifulSoup


# Function to extract text
def extract_text(soup):
    for script in soup(["script", "style"]):
        script.decompose()  # Remove script and style elements
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


# Function to extract images
def extract_images(soup, base_url):
    images = []
    for img in soup.find_all('img'):
        img_url = img.get('src')
        if not img_url.startswith('http'):
            img_url = base_url + img_url
        images.append(img_url)
    return images


def get_url_text_and_images(url):
    # Fetch the content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text
    text = extract_text(soup)
    # Extract images
    images = extract_images(soup, url)
    return text, images


def is_valid_image_url(url):
    try:
        response = requests.head(url)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            if content_type and 'image' in content_type:
                return True
        return False
    except requests.RequestException:
        return False


def get_valid_image_urls(urls):
    valid_image_urls = []
    for url in urls:
        if is_valid_image_url(url):
            valid_image_urls.append(url)
    return valid_image_urls


def filter_invalid_image_paths(image_paths):
    valid_image_paths = []
    exclusion_components = ["&noscript", "facebook", "instagram", "twitter", "youtube", "icon", "linkedin"]
    valid_file_endings = [".png", ".jpg", ".jpeg", ".gif"]
    for image_path in image_paths:
        found = [s for s in exclusion_components if s in image_path]
        if len(found) == 0:
            found = [s for s in valid_file_endings if s in image_path]
            if len(found) == 1:
                index_target = len(image_path) - len(found[0])
                found_index = image_path.index(found[0])
                if found_index == index_target:
                    valid_image_paths.append(image_path)
    return list(set(valid_image_paths))


def save_images(images):
    os.makedirs('images', exist_ok=True)
    for img_url in images:
        img_response = requests.get(img_url, stream=True)
        if img_response.status_code == 200:
            img_name = os.path.join('images', os.path.basename(img_url))
            with open(img_name, 'wb') as f:
                for chunk in img_response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved {img_name}")
        else:
            print("Could not get image for: ", img_url)


if __name__ == '__main__':
    # URL of the website
    url = 'https://zugvogel.design/'
    text, image_paths = get_url_text_and_images(url)
    print("Text content:\n", text)
    print("Image URLs:\n", image_paths)
    valid_image_paths = filter_invalid_image_paths(image_paths)
    valid_image_paths = get_valid_image_urls(valid_image_paths)
    # Optionally, download images
    save_images(valid_image_paths)

    with open("./src/config/keys.json") as f:
        config = json.load(f)
    openai_api_key = config["openai_api_key"]

    client = OpenAI(api_key=openai_api_key)

    for valid_image_path in valid_image_paths:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": valid_image_path,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print(response.choices[0])
