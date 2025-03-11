import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
import requests
from PIL import Image
import io
import base64


class ImageProcessor:
    def __init__(self, model="meta/llama-3.2-11b-vision-instruct", base_url_nim="http://127.0.0.1:8000/v1"):
        self.llm = ChatNVIDIA(model=model, base_url=base_url_nim)  # ✅ Use multimodal base URL
        self.invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
        self.stream = False
        self.image_pattern = re.compile(r'!\[\]\((.*?)\)')

    def process_images(self, markdown_text):
        """Finds images in Markdown, sends them to NIM for summarization."""
        def add_metadata(match):
            image_path = match.group(1)
            # summary = ""
            summary = self.generate_summary(image_path, markdown_text)
            return f'<!-- image: {image_path} summary: "{summary}" -->'

        return self.image_pattern.sub(add_metadata, markdown_text)

    def generate_summary(self, img_path, page_text):
        """Calls NVIDIA NIM to generate an image summary using page text as context."""

        image_b64 = img2base64_string(img_path)
        
        assert len(image_b64) < 180_000, \
            "To upload larger images, use the assets API (see docs)"
        
        headers = {
            "Authorization": "Bearer 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",
            "Accept": "text/event-stream" if self.stream else "application/json"
        }

        payload = {
            "model": 'meta/llama-3.2-11b-vision-instruct',
            "messages": [
                {
                "role": "user",
                "content": f'Summarize this image, here is the rest of the text on the page for better context: {page_text}? <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 1.00,
            "top_p": 1.00,
            "stream": self.stream
        
        }

        response = requests.post(self.invoke_url, headers=headers, json=payload)

        lines= []

        if self.stream:
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8")['data']['choices']['delta']['content'])
            return lines
        else:
            obj = response.json()
            return obj['choices'][0]['message']['content']


def img2base64_string(img_path):
    image = Image.open(img_path)
    if image.width > 800 or image.height > 800:
        image.thumbnail((800, 800))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return image_base64