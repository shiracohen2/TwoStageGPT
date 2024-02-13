import os
from dotenv import load_dotenv

from dataclasses import dataclass, field
from conf.base_gpt_config import BaseGptConfig

load_dotenv()


@dataclass
class Gpt4VisionConfig(BaseGptConfig):
    api_key: str = os.getenv("GPT4_VISION_KEY")
    azure_endpoint: str = os.getenv("GPT4_VISION_ENDPOINT")
    vision_model_deployment_name: str = os.open("GPT4_VISION_DEPLOYMENT_NAME")
