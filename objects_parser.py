from logging import Logger
from pathlib import Path

from tqdm import tqdm

from conf.gpt4_lang_config import GPT4LangConfig
from conf.data_config import DataConfig
from base_gpt_clevr_solver import BaseGptClevrSolver
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.gpt4_lang_client import Gpt4LangClient
from utils.logger import init_logger


class ObjectsParser(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4LangClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4LangClient = gpt_client
        self.one_step_gpt_results_file: Path = data_config.one_step_gpt_results_file
        self.objects_parsing_results_file: Path = data_config.objects_parsing_results_file

    @property
    def prompt(self) -> str:
        prompt = ("Identify and list all objects mentioned in the following <question>.\n"
                  "Your response should consist solely of this list, with each object clearly enumerated. "
                  "Do not enumerate objects with their counts, only their descriptions. \n"
                  "For example, if the <question> is: 'Add 5 blue balls. Add 2 balls. How many objects exist?',"
                  "your response should be: 'blue balls, balls'.\n"
                  "Another example: if the <question> is: 'Add 5 small objects. Add 2 metal objects. "
                  "How many balls are there?',"
                  "your response should be: 'small objects, metal objects, balls'.\n\n"
                  "<question>: {question}")
        return prompt

    def parse_questions(self) -> dict[str, dict]:
        """
        Iterate over the questions and parse the objects.
        Save the results in a json file.
        """
        self.logger.info("Starting questions parsing")
        results: dict[str, dict] = {}

        try:
            one_step_gpt_results = self.load_json_file(self.one_step_gpt_results_file)
            for question_index, question_result in tqdm(one_step_gpt_results.items()):
                question_parsing_result = self.get_question_parsing_result(
                    question_data=question_result,
                )

                results[question_index] = question_parsing_result

        except Exception as e:
            self.logger.error(f"Failed to parse questions: {e}")
            raise e

        finally:
            return results

    def get_question_parsing_result(self, question_data) -> dict[str, str]:
        """
        Get the question parsing result from the gpt.
        """
        # get question and prompt and send to gpt
        question = question_data[ImageDataEnum.QUESTION]
        prompt = self.prompt.format(question=question)
        question_parsing_result = self.gpt_client.get_lang_model_response(prompt=prompt)
        result = {
            ImageDataEnum.IMAGE_PATH: question_data[ImageDataEnum.IMAGE_PATH],
            ImageDataEnum.IMAGE_ID: question_data[ImageDataEnum.IMAGE_ID],
            ImageDataEnum.QUESTION: question,
            ImageDataEnum.TEMPLATE: question_data[ImageDataEnum.TEMPLATE],
            ImageDataEnum.LABEL: question_data[ImageDataEnum.LABEL],
            ImageDataEnum.PARSING_RESULT: question_parsing_result,

        }
        return result


if __name__ == "__main__":
    logger = init_logger(file_name="objects_parser.log")
    gpt_client = Gpt4LangClient(config=GPT4LangConfig(), logger=logger)

    config = DataConfig()
    objects_parser = ObjectsParser(data_config=config, gpt_client=gpt_client, logger=logger)
    objects_parsing_results = objects_parser.parse_questions()
    objects_parser.save_json_file(file_path=objects_parser.objects_parsing_results_file,
                                  data=objects_parsing_results)
