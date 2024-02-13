from logging import Logger
from pathlib import Path
from typing import Any

from tqdm import tqdm

from conf.gpt4_lang_config import GPT4LangConfig
from conf.data_config import DataConfig
from base_gpt_clevr_solver import BaseGptClevrSolver
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.gpt4_lang_client import Gpt4LangClient
from utils.logger import init_logger


class TwoStepGpt(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4LangClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4LangClient = gpt_client
        self.two_step_gpt_results_file: Path = data_config.two_step_gpt_results_file
        self.object_counting_results_file: Path = data_config.object_counting_results_file

    @property
    def prompt(self) -> str:
        prompt = ("You are given a question about an image, without the image itself.\n"
                  "You are also given a description of the image.\n"
                  "Answer the <question>, based on the <description>.\n"
                  "Your response should include not only the numerical answer but also a brief explanation of how "
                  "you arrived at that conclusion.\n"
                  "Pay attention: make sure to conclude your answer with the following format: \n"
                  "'My answer is: <numeric answer>'\n"
                  "For e.g.: 'My answer is: 64'\n\n"
                  "<question>: {question}\n"
                  "<description>:\n{description}")
        return prompt

    def solve_questions(self) -> dict[int, dict]:
        results = {}
        self.logger.info("Starting solving questions")
        counting_results = self.load_json_file(file_path=self.object_counting_results_file)

        try:
            for question_index, counting_data in tqdm(counting_results.items()):
                count_res: str = counting_data[ImageDataEnum.COUNTING_RESULT]
                parsing_res: str = counting_data[ImageDataEnum.PARSING_RESULT]

                question_result = self.get_question_result(
                    question_data=counting_data,
                    parsing_result=parsing_res,
                    counting_result=count_res

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
            parsing_result: str,
            counting_result: str
    ) -> dict[str, str]:
        # prepare the data for the gpt model and get the response
        image_path = question_data[ImageDataEnum.IMAGE_PATH]
        question = question_data[ImageDataEnum.QUESTION]
        prompt = self.prompt.format(question=question, description=counting_result)
        gpt_response = self.gpt_client.get_lang_model_response(prompt)

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
            ImageDataEnum.COUNTING_RESULT: counting_result,
            ImageDataEnum.GPT_RESPONSE: gpt_response,
            ImageDataEnum.NUMERICAL_RESULT: numerical_result,
            ImageDataEnum.IS_CORRECT: is_correct,
        }
        return result


if __name__ == "__main__":
    logger = init_logger(file_name="two_step_gpt.log")

    gpt_config = GPT4LangConfig()
    gpt_client = Gpt4LangClient(config=gpt_config, logger=logger)

    config = DataConfig()
    two_step_gpt = TwoStepGpt(data_config=config, gpt_client=gpt_client, logger=logger)

    answers = two_step_gpt.solve_questions()
    logger.info(f"Finished solving questions.")
    two_step_gpt.save_json_file(file_path=two_step_gpt.two_step_gpt_results_file, data=answers)
    logger.info(f"Saved results to {two_step_gpt.two_step_gpt_results_file}")
