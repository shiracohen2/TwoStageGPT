from logging import Logger
from pathlib import Path
from tqdm import tqdm

from conf.data_config import DataConfig
from conf.gpt_4_vision_config import Gpt4VisionConfig
from data_enums.image_data_enum import ImageDataEnum
from experiments.base_gpt_clevr_solver import BaseGptClevrSolver
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from utils.logger import init_logger


class SimpleObjectDetector(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4VisionClient = gpt_client
        self.simple_object_detection_results_file: Path = data_config.simple_object_detection_results_file
        self.one_step_gpt_results_file: Path = data_config.one_step_gpt_results_file
        self.sampled_questions: Path = data_config.sampled_keys_for_validation

    @property
    def prompt(self) -> str:
        prompt = (
            "Describe and count the objects in the image.\n"
            "In your response, describe the objects by the following attributes: "
            "color, size (small/large), shape(sphere/cube/cylinder) and material(shiny/matte).\n"
            "Conclude your answer with the total number of objects in the image.\n"
            "For example: There are 2 large shiny spheres, 1 small shiny cube, 1 large matte cylinder, "
            "and 4 objects in total."
        )
        return prompt

    def solve_questions(self) -> dict[int, dict]:
        """
        Iterate over the images from the test experiment and solve the questions.
        Save the results in a json file.
        """
        self.logger.info("Starting questions solving")
        results = {}

        try:
            one_step_gpt_results = self.load_json_file(self.one_step_gpt_results_file)
            sampled_keys = self.load_sampled_keys_list(self.sampled_questions)

            for question_index, question_data in tqdm(one_step_gpt_results.items()):
                question_index = int(question_index)
                if question_index not in sampled_keys:
                    continue
                image_path = question_data[ImageDataEnum.IMAGE_PATH]
                detection_result = self.detect_objects(image_path=image_path)
                results[question_index] = {
                    "detection_result": detection_result,
                    "number_of_objects": question_data.get("number_of_objects", None),
                    ImageDataEnum.IMAGE_PATH: image_path,
                    ImageDataEnum.IMAGE_ID: question_data[ImageDataEnum.IMAGE_ID],
                    ImageDataEnum.QUESTION: question_data[ImageDataEnum.QUESTION]
                }

        except Exception as e:
            self.logger.exception(f"Error while solving questions: {e}")
            raise e
        finally:
            return results

    def detect_objects(self, image_path: str) -> str:
        """
        Call the GPT model to solve the question and return the result.
        """
        # prepare the data for the gpt model and get the response
        gpt_response = self.gpt_client.get_vision_model_response(image_path, self.prompt)

        return gpt_response

    def load_sampled_keys_list(self, sampled_questions) -> set[int]:
        """
        Read the sampled keys from the file: Each line contains a key (int).
        Save the keys in a set and return it.
        """
        sampled_keys = set()
        with open(sampled_questions, "r") as f:
            for line in f:
                sampled_keys.add(int(line.strip()))
        return sampled_keys


if __name__ == "__main__":
    logger = init_logger(file_name="simple_object_detector.log")

    gpt_config = Gpt4VisionConfig()
    gpt_vision_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    detector = SimpleObjectDetector(data_config=config, gpt_client=gpt_vision_client, logger=logger)
    answers = detector.solve_questions()

    detector.save_json_file(file_path=detector.simple_object_detection_results_file, data=answers)
    logger.info(f"Finished detecting objects.")
    logger.info(f"Results saved in {detector.simple_object_detection_results_file}")
