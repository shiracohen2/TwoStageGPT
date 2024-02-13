from logging import Logger
from pathlib import Path
from typing import Any

from tqdm import tqdm

from conf.gpt_4_vision_config import Gpt4VisionConfig
from conf.data_config import DataConfig
from data_enums.clevr_descriptions_enum import ClevrDescriptionsEnum
from data_enums.clevr_math_labels_enum import ClevrMathLabelsEnum
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from utils.logger import init_logger
from one_step_gpt import OneStepGPT


class OracleOneStep(OneStepGPT):
    """
    This class is used to solve the questions from CLEVR-math in a one-step approach.
    However, in this case, in addition to the image and the question, the model also receives a full
    description of all the objects in the image (taken from CLEVR dataset annotations).

    """
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.oracle_one_step_results_file: Path = data_config.oracle_one_step_results_file
        self.one_step_gpt_results_file: Path = Path(__file__).parent.joinpath(
            "data",
            "validation_set_results",
            "one_step_gpt_results.json"
        )

    @property
    def prompt(self) -> str:
        prompt = ("Answer the following <question>, based on the given image. You are also provided with a description "
                  "of all the objects present in the image, which can assist you in solving the question.\n"
                  "Your response should include not only the numerical answer but also a brief explanation of how "
                  "you arrived at that conclusion.\n"
                  "Pay attention: make sure to conclude your answer with the following format: \n"
                  "'My answer is: <numeric answer>'\n"
                  "For e.g.: 'My answer is: 64'\n\n"
                  "<description>: {description}\n\n"
                  "<question>: {question}")
        return prompt

    def solve_questions(self) -> dict[int, dict]:
        """
        Iterate over the images from the experiment and solve the questions.
        Save the results in a json file.
        """
        self.logger.info("Starting questions solving")
        results = {}

        try:
            one_step_gpt_results = self.load_json_file(self.one_step_gpt_results_file)
            dataset = self.download_dataset(self.clevr_math_dataset_name)[ClevrMathLabelsEnum.CHOSEN_DATASET]
            clevr_val_scenes = self.load_json_file(file_path=self.clevr_val_scenes)[ClevrDescriptionsEnum.SCENES]

            for question_index in tqdm(one_step_gpt_results.keys()):
                question_index = int(question_index)
                question_data = dataset[question_index]
                template = question_data[ClevrMathLabelsEnum.TEMPLATE]

                # Get the image scene
                image_id = question_data[ClevrMathLabelsEnum.ID]
                image_index = self.get_image_index_from_id(image_id=image_id)
                image_data = clevr_val_scenes[image_index]
                image_scene = image_data[ClevrDescriptionsEnum.OBJECTS]

                question_result = self.get_question_result_with_scenes(question_data=question_data, image_scene=image_scene)
                results[question_index] = question_result
                # update counters
                self.questions_counter[template] += 1

        except Exception as e:
            self.logger.exception(f"Error while solving questions: {e}")
            raise e
        finally:
            return results

    def get_question_result_with_scenes(self, question_data: dict[str, Any], image_scene: dict) -> dict[str, str]:
        image_path = question_data[ClevrMathLabelsEnum.IMAGE].filename
        question = question_data[ClevrMathLabelsEnum.QUESTION]

        description = "The image contains the following objects:\n"
        for data in image_scene:
            description += f"{data['size']} {data['color']} {data['material']} {data['shape']}\n"

        prompt = self.prompt.format(question=question, description=description)
        gpt_response = self.gpt_client.get_vision_model_response(image_path, prompt)

        return self.create_result(
            gpt_response=gpt_response,
            image_path=image_path,
            question=question,
            question_data=question_data
        )


if __name__ == "__main__":
    logger = init_logger(file_name="oracle_one_step_gpt.log")

    gpt_config = Gpt4VisionConfig()
    gpt_vision_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    oracle_one_step_gpt_cot = OracleOneStep(data_config=config, gpt_client=gpt_vision_client, logger=logger)
    answers = oracle_one_step_gpt_cot.solve_questions()

    oracle_one_step_gpt_cot.save_json_file(file_path=oracle_one_step_gpt_cot.oracle_one_step_results_file, data=answers)
    logger.info(f"Finished solving questions.")
    logger.info(f"Number of correct answers: {oracle_one_step_gpt_cot.get_number_of_correct_answers(results=answers)}")
