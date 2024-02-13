import logging
import time
from abc import ABC

from openai import RateLimitError
from openai.lib.azure import AzureOpenAI

from conf.base_gpt_config import BaseGptConfig


class BaseClient(ABC):
    def __init__(self, config: BaseGptConfig, logger: logging.Logger):
        self.deployment_name = config.vision_model_deployment_name
        self.max_rate_limit_retries = config.max_rate_limit_retries
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.logger = logger
        self.client: AzureOpenAI = AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version
        )

    def handle_rate_limit_error(self):
        wait_time = 5
        self.logger.info(f"Rate limit error encountered. Waiting for {wait_time} seconds.")
        time.sleep(wait_time)

    def _get_response(self, messages: list[dict]):
        rate_limit_error_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if rate_limit_error_count >= self.max_rate_limit_retries:
                    self.logger.error(f"Rate limit error count exceeded {self.max_rate_limit_retries}. Exiting.")
                    raise e
                else:
                    self.handle_rate_limit_error()
                    rate_limit_error_count += 1
                    continue


