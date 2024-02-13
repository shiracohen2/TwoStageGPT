from collections import Counter
from logging import Logger
from pathlib import Path
from random import randint
from typing import Any

from tqdm import tqdm

from data_enums.clevr_math_labels_enum import ClevrMathLabelsEnum
from conf.gpt_4_vision_config import Gpt4VisionConfig
from conf.data_config import DataConfig
from data_enums.image_data_enum import ImageDataEnum
from experiments.base_gpt_clevr_solver import BaseGptClevrSolver
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from utils.logger import init_logger


class OneStepGPT(BaseGptClevrSolver):
    """
    This class is used to solve the questions from CLEVR-math in a one-step approach:
    the model receives an image and a question and should provide the answer.
    """
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4VisionClient = gpt_client
        self.one_step_gpt_results_file: Path = data_config.one_step_gpt_results_file
        self.number_of_questions_to_solve: int = data_config.number_of_questions_to_solve
        self.questions_counter: Counter = Counter()
        self.limit_question_type: int = self.number_of_questions_to_solve // 4

    @property
    def prompt(self) -> str:
        """
        The prompt used to get the model response.
        """
        prompt = ("Answer the following <question>, based on the given image.\n"
                  "Your response should include not only the numerical answer but also a brief explanation of how "
                  "you arrived at that conclusion.\n"
                  "Pay attention: make sure to conclude your answer with the following format: \n"
                  "'My answer is: <numeric answer>'\n"
                  "For e.g.: 'My answer is: 64'\n\n"
                  "<question>: {question}")
        return prompt

    def solve_questions(self) -> dict[int, dict]:
        """
        Iterate over the images and solve the questions.
        Save the results in a json file.
        """
        self.logger.info("Starting questions solving")
        results = {}

        try:
            questions_solved = 0
            dataset = self.download_dataset(self.clevr_math_dataset_name)[ClevrMathLabelsEnum.CHOSEN_DATASET]
            # rand index to start from
            progress_bar = tqdm(total=self.number_of_questions_to_solve - questions_solved)

            while questions_solved < self.number_of_questions_to_solve:
                i = randint(0, len(dataset) - 1)
                question_data = dataset[i]
                template = question_data[ClevrMathLabelsEnum.TEMPLATE]
                if self.questions_counter[template] >= self.limit_question_type or i in results:
                    continue

                question_result = self.get_question_result(question_data=question_data)
                results[i] = question_result
                # update counters
                self.questions_counter[template] += 1
                questions_solved += 1
                progress_bar.update(1)

        finally:
            return results

    def get_question_result(self, question_data: dict[str, Any]) -> dict[str, str]:
        """
        Call the GPT model to solve the question and return the result.
        """
        # prepare the data for the gpt model and get the response
        image_path = question_data[ClevrMathLabelsEnum.IMAGE].filename
        question = question_data[ClevrMathLabelsEnum.QUESTION]
        prompt = self.prompt.format(question=question)
        gpt_response = self.gpt_client.get_vision_model_response(image_path, prompt)

        return self.create_result(
            gpt_response=gpt_response,
            image_path=image_path,
            question=question,
            question_data=question_data
        )

    def create_result(self, gpt_response, image_path, question, question_data):
        template = question_data[ClevrMathLabelsEnum.TEMPLATE]
        image_id = question_data[ClevrMathLabelsEnum.ID]
        label = question_data[ClevrMathLabelsEnum.LABEL]
        numerical_result = self.extract_numeric_answer(text=gpt_response)
        is_correct = label == numerical_result
        result = {
            ImageDataEnum.IMAGE_PATH: image_path,
            ImageDataEnum.IMAGE_ID: image_id,
            ImageDataEnum.QUESTION: question,
            ImageDataEnum.TEMPLATE: template,
            ImageDataEnum.LABEL: label,
            ImageDataEnum.GPT_RESPONSE: gpt_response,
            ImageDataEnum.NUMERICAL_RESULT: numerical_result,
            ImageDataEnum.IS_CORRECT: is_correct,
        }
        return result

    def get_model_response(self, image_path: str, prompt: str) -> str:
        """
        Get the model response for a given image and prompt.
        """
        return self.gpt_client.get_vision_model_response(image_path, prompt)


if __name__ == "__main__":
    logger = init_logger(file_name="one_step_gpt.log")

    gpt_config = Gpt4VisionConfig()
    gpt_vision_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    one_step_gpt = OneStepGPT(data_config=config, gpt_client=gpt_vision_client, logger=logger)
    answers = one_step_gpt.solve_questions()

    one_step_gpt.save_json_file(file_path=one_step_gpt.one_step_gpt_results_file, data=answers)
    print(one_step_gpt.questions_counter)
