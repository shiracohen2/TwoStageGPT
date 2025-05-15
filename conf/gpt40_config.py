import os
from dotenv import load_dotenv

from dataclasses import dataclass, field
from conf.base_gpt_config import BaseGptConfig

load_dotenv()


@dataclass
class GPT4OConfig(BaseGptConfig):
    api_key: str = os.getenv("GPT4O_KEY")
    azure_endpoint: str = os.getenv("GPT4O_ENDPOINT")
    vision_model_deployment_name: str = os.getenv("GPT4O_DEPLOYMENT_NAME")
    api_version: str = field(default="2025-01-01-preview")
