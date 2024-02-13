from logging import Logger
from pathlib import Path
from typing import Any

from tqdm import tqdm

from base_gpt_clevr_solver import BaseGptClevrSolver
from conf.gpt_4_vision_config import Gpt4VisionConfig
from conf.data_config import DataConfig
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from logger import init_logger


class OracleTwoStep(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4VisionClient = gpt_client
        self.oracle_two_step_results_file: Path = data_config.oracle_two_step_results_file
        self.oracle_parsing_results_file: Path = data_config.oracle_parsing_results_file

    @property
    def prompt(self) -> str:
        prompt = (
            "You are given a <question> about an image.\n"
            "You are also given a <description> of the objects in the image that are relevant to "
            "the the <question>.\n"
            "Answer the <question>.\n"
            "Your response should include not only the numerical answer but also a brief explanation of how "
            "you arrived at that conclusion.\n"
            "Pay attention: make sure to conclude your answer with the following format: \n"
            "'My answer is: <numeric answer>'\n"
            "For e.g.: 'My answer is: 64'\n\n"
            "<question>: {question}\n"
            "<description>:\n{description}"
        )
        return prompt

    def solve_questions(self) -> dict[int, dict]:
        results = {}
        self.logger.info("Starting solving questions")
        parsing_results = self.load_json_file(file_path=self.oracle_parsing_results_file)

        try:
            for question_index, question_data in tqdm(parsing_results.items()):
                parsing_res: str = question_data[ImageDataEnum.PARSING_RESULT]
                if parsing_res is None:
                    parsing_res = ""

                question_result = self.get_question_result(
                    question_data=question_data,
                    parsing_result=parsing_res,

                )
                results[question_index] = question_result

        except Exception as e:
            self.logger.exception(f"Failed to count objects. Error: {e}")
            raise e
        finally:
            return results

    def get_question_result(
            self,
            question_data: dict[str, Any],
            parsing_result: str
    ) -> dict[str, str]:
        # prepare the data for the gpt model and get the response
        image_path = question_data[ImageDataEnum.IMAGE_PATH]
        question = question_data[ImageDataEnum.QUESTION]
        prompt = self.prompt.format(question=question, description=parsing_result)
        gpt_response = self.gpt_client.get_vision_model_response(prompt=prompt, image_path=image_path)

        template = question_data[ImageDataEnum.TEMPLATE]
        image_id = question_data[ImageDataEnum.IMAGE_ID]
        label = question_data[ImageDataEnum.LABEL]
        numerical_result = self.extract_numeric_answer(text=gpt_response)
        is_correct = label == numerical_result

        result = {
            ImageDataEnum.IMAGE_PATH: image_path,
            ImageDataEnum.IMAGE_ID: image_id,
            ImageDataEnum.QUESTION: question,
            ImageDataEnum.TEMPLATE: template,
            ImageDataEnum.LABEL: label,
            ImageDataEnum.PARSING_RESULT: parsing_result,
            ImageDataEnum.GPT_RESPONSE: gpt_response,
            ImageDataEnum.NUMERICAL_RESULT: numerical_result,
            ImageDataEnum.IS_CORRECT: is_correct,
        }
        return result


if __name__ == "__main__":
    logger = init_logger(file_name="oracle_two_step.log")

    gpt_config = Gpt4VisionConfig()
    gpt_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    oracle_two_step = OracleTwoStep(data_config=config, gpt_client=gpt_client, logger=logger)

    answers = oracle_two_step.solve_questions()
    logger.info(f"Finished solving questions.")
    oracle_two_step.save_json_file(file_path=oracle_two_step.oracle_two_step_results_file, data=answers)
    logger.info(f"Number of correct answers: {oracle_two_step.get_number_of_correct_answers(results=answers)}")
