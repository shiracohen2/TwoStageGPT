from gpt_clients.base_client import BaseClient


class Gpt4LangClient(BaseClient):
    def get_lang_model_response(self, prompt: str) -> str:
        messages = self.prepare_messages(prompt=prompt)
        response = self._get_response(messages=messages)
        return response

    def prepare_messages(self, prompt: str) -> list[dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        return messages
