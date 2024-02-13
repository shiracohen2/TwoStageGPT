import os
from dotenv import load_dotenv

from dataclasses import dataclass
from conf.base_gpt_config import BaseGptConfig

load_dotenv()


@dataclass
class GPT4LangConfig(BaseGptConfig):
    api_key: str = os.getenv("GPT4_LANG_KEY")
    azure_endpoint: str = os.getenv("GPT4_LANG_ENDPOINT")
    vision_model_deployment_name: str = os.getenv("GPT4_LANG_DEPLOYMENT_NAME")
