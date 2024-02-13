from logging import Logger
from pathlib import Path

from tqdm import tqdm

from conf.gpt4_lang_config import GPT4LangConfig
from conf.data_config import DataConfig
from base_gpt_clevr_solver import BaseGptClevrSolver
from data_enums.clevr_descriptions_enum import ClevrDescriptionsEnum
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.gpt4_lang_client import Gpt4LangClient
from logger import init_logger


class OracleObjectsParser(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4LangClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4LangClient = gpt_client
        self.oracle_one_step_results_file: Path = data_config.oracle_one_step_results_file
        self.oracle_parsing_results_file: Path = data_config.oracle_parsing_results_file

    @property
    def prompt(self) -> str:
        prompt = (
            "You are provided with a <description> of an image. This description describes all the objects present in "
            "the image.\n"
            "You are also given a <question> about that image. You task is to return back all the objects from the "
            "description that are mentioned in the <question>.\n"

            "For example, if the <description> is:\n"
            "'The image contains the following objects:\n"
            "large red metal sphere\n"
            "small blue rubber cube\n"
            "large green metal cylinder\n"
            "small red rubber sphere\n'\n"

            "And the <question> is:\n"
            "'Subtract one red ball. How many red balls are left in the image?'\n"

            "Then your response should contain only the lines from the <description> that contain red balls:\n"
            "large red metal sphere\n"
            "small red rubber sphere\n'\n"

            "Another example: given the same description from the example above, and the following <question>:"
            "Add one green object. How many object are in the image?\n"
            "Then your response should be:\n"
            "'large red metal sphere\n"
            "small blue rubber cube\n"
            "large green metal cylinder\n"
            "small red rubber sphere\n'"
            
            "Pay attention: your answer should include only the relevant lines from the <description> and nothing "
            "else. If you can't find the objects from the question in the description return an empty string.\n\n"

            "<question>: {question}"
            "<description>: {description}"
        )
        return prompt

    def parse_questions(self) -> dict[str, dict]:
        """
        Iterate over the questions and parse the objects.
        Save the results in a json file.
        """
        self.logger.info("Starting questions parsing")
        results: dict[str, dict] = {}

        try:
            oracle_one_step_results = self.load_json_file(self.oracle_one_step_results_file)
            clevr_val_scenes = self.load_json_file(file_path=self.clevr_val_scenes)[ClevrDescriptionsEnum.SCENES]
            for question_index, question_result in tqdm(oracle_one_step_results.items()):
                image_id = question_result[ImageDataEnum.IMAGE_ID]
                image_index = self.get_image_index_from_id(image_id=image_id)
                image_data = clevr_val_scenes[image_index]
                image_scene = image_data[ClevrDescriptionsEnum.OBJECTS]
                question_parsing_result = self.get_question_parsing_result(
                    question_data=question_result,
                    image_scene=image_scene
                )

                results[question_index] = question_parsing_result

        except Exception as e:
            self.logger.error(f"Failed to parse questions: {e}")
            raise e

        finally:
            return results

    def get_question_parsing_result(self, question_data, image_scene: dict) -> dict[str, str]:
        """
        Get the question parsing result.
        """
        # get question and prompt and send to gpt
        question = question_data[ImageDataEnum.QUESTION]

        description = "The image contains the following objects:\n"
        for data in image_scene:
            description += f"{data['size']} {data['color']} {data['material']} {data['shape']}\n"

        prompt = self.prompt.format(question=question, description=description)

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
    logger = init_logger(file_name="oracle_parser.log")
    gpt_client = Gpt4LangClient(config=GPT4LangConfig(), logger=logger)

    config = DataConfig()
    oracle_parser = OracleObjectsParser(data_config=config, gpt_client=gpt_client, logger=logger)
    objects_parsing_results = oracle_parser.parse_questions()
    oracle_parser.save_json_file(
        file_path=oracle_parser.oracle_parsing_results_file,
        data=objects_parsing_results
    )
