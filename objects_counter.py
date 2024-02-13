from logging import Logger
from pathlib import Path

from tqdm import tqdm

from conf.gpt_4_vision_config import Gpt4VisionConfig
from conf.data_config import DataConfig
from base_gpt_clevr_solver import BaseGptClevrSolver
from data_enums.image_data_enum import ImageDataEnum
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from utils.logger import init_logger


class ObjectsCounter(BaseGptClevrSolver):
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.gpt_client: Gpt4VisionClient = gpt_client
        self.objects_parsing_results_file: Path = data_config.objects_parsing_results_file
        self.object_counting_results_file: Path = data_config.object_counting_results_file

    @property
    def prompt(self) -> str:
        prompt = (
            "Analyze the provided image and identify the objects from the specified <objects list>.\n"
            "Your task is to count the number of each listed object present in the image. "
            "Ensure that your counts are accurate and consider any overlaps or subsets among the object "
            "categories. Remember to distinguish between specific and general categories "
            "(e.g., 'red balls' as a subset of 'balls').\n"
            "Pay careful attention to the following:\n"
            "1. The <objects list> may include objects that are not present in the image.\n"
            "2. The <objects list> may not include all the objects present in the image.\n"
            "3. Structure your response as an ordered list, with each entry including the name of the object "
            "from the <objects list>, followed by a short description of its appearances in the image, "
            "referring their color, size(small or large), shape(cube, ball, cylinder, etc.), "
            "material(matte/rubber or metal/shiny), and finally their total count.\n"
            "Only when you are required to count all the objects in the image, simply provide the total "
            "count - without the descriptions. You should always add this count to the end of your response.\n"
            "For example, if the <objects list> is: 'red balls, balls, small objects, cylinders', "
            "your response should look like:\n"
            "'1. red balls: 1 small metal/shiny red ball, 1 large red rubber/matte ball, 1 large red "
            "metal/shiny ball. Total: 3\n"
            "2. balls: 1 small metal/shiny red ball, 1 large red rubber/matte ball, 1 large red metal/shiny "
            "ball, 1 small purple rubber/matte ball. Total: 4\n"
            "3. large objects: 1 large blue metal/shiny cube, 1 large red rubber/matte ball. Total: 2\n"
            "4. cylinders: Not present in the image. Total: 0\n"
            "5. objects: 10'\n\n"
            "<objects list>: {objects_list}"
        )

        return prompt

    def count_objects(self) -> dict[int, dict]:
        self.logger.info("Starting objects counting")
        results = {}
        try:
            parsing_results = self.load_json_file(file_path=self.objects_parsing_results_file)
            for question_index, question_data in tqdm(parsing_results.items()):
                parsing_result: str = question_data[ImageDataEnum.PARSING_RESULT]

                counting_result = self.get_counting_result(
                    question_data=question_data,
                    parsing_result=parsing_result
                )
                results[question_index] = counting_result

        except Exception as e:
            self.logger.error(f"Failed to count objects. Error: {e}")
            raise e
        finally:
            return results

    def get_counting_result(self, question_data: dict, parsing_result: str) -> dict[str, str]:
        """
        Get the counting result from the model.
        """
        image_path = question_data[ImageDataEnum.IMAGE_PATH]
        prompt = self.prompt.format(objects_list=parsing_result)
        gpt_response = self.gpt_client.get_vision_model_response(prompt=prompt, image_path=image_path)

        image_id = question_data[ImageDataEnum.IMAGE_ID]
        question = question_data[ImageDataEnum.QUESTION]
        template = question_data[ImageDataEnum.TEMPLATE]
        label = question_data[ImageDataEnum.LABEL]

        result = {
            ImageDataEnum.IMAGE_PATH: image_path,
            ImageDataEnum.IMAGE_ID: image_id,
            ImageDataEnum.QUESTION: question,
            ImageDataEnum.TEMPLATE: template,
            ImageDataEnum.LABEL: label,
            ImageDataEnum.PARSING_RESULT: parsing_result,
            ImageDataEnum.COUNTING_RESULT: gpt_response,
        }

        return result


if __name__ == "__main__":
    logger = init_logger(file_name="objects_counter.log")

    gpt_config = Gpt4VisionConfig()
    gpt_vision_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    objects_counter = ObjectsCounter(data_config=config, gpt_client=gpt_vision_client, logger=logger)
    answers = objects_counter.count_objects()
    logger.info(f"Finished objects counting. Saving results.")

    objects_counter.save_json_file(file_path=objects_counter.object_counting_results_file, data=answers)
    logger.info(f"Finished objects counting. Results saved in {objects_counter.object_counting_results_file}")
