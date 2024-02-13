import json
import re

from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Union

from datasets import load_dataset, DownloadConfig
from conf.data_config import DataConfig
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.base_client import BaseClient


class BaseGptClevrSolver(ABC):

    def __init__(self, data_config: DataConfig, gpt_client: BaseClient, logger: Logger):
        self.logger = logger
        self.gpt_client = gpt_client
        self.clevr_val_scenes: Path = data_config.clevr_val_scenes
        self.clevr_math_dataset_name = data_config.clevr_math_dataset_name

    @property
    @abstractmethod
    def prompt(self) -> str:
        raise NotImplementedError

    @staticmethod
    def download_dataset(dataset_name: str):
        dl_config = DownloadConfig(resume_download=True,
                                   num_proc=8,
                                   force_download=True)
        dataset = load_dataset(path=dataset_name, download_config=dl_config, trust_remote_code=True)
        return dataset

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json_file(file_path, data):
        with open(file_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def get_image_index_from_id(image_id: str) -> int:
        """
        Given an image id, return the image index.
        Example of image_id: CLEVR_val_000000.png
        Corresponding image_index: 0
        """
        return int(image_id.split(".")[0].split("_")[-1])

    @staticmethod
    def extract_numeric_answer(text) -> Union[int, None]:
        """
        Extracts the numeric answer from a text using regex.

        Args:
        text (str): The text containing the numeric answer in the format 'My answer is: <numeric answer>'

        Returns:
        int or None: The extracted numeric answer, or None if not found.
        """
        match = re.search(r"My answer is: (\d+)", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def get_number_of_correct_answers(results: dict[int, dict]) -> int:
        """
        Get the number of correct answers from the results.
        """
        return sum(
            1 for question_index, question_result in results.items()
            if question_result.get(ImageDataEnum.IS_CORRECT) is True
        )
