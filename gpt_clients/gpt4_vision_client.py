import base64
from typing import Optional

from gpt_clients.base_client import BaseClient


class Gpt4VisionClient(BaseClient):

    def get_vision_model_response(
            self,
            image_path: str,
            prompt: str,
            chain_of_thought_messages: Optional[list[dict]] = None
    ) -> str:
        if chain_of_thought_messages is None:
            chain_of_thought_messages = []
        prompt_messages = self.prepare_messages(image_path=image_path, prompt=prompt)
        messages = chain_of_thought_messages + prompt_messages
        response = self._get_response(messages=messages)
        return response

    def prepare_messages(self, image_path: str, prompt: str) -> list[dict]:
        base64_image = self.encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            },
        ]

        return messages

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
