from logging import Logger
from pathlib import Path
from tqdm import tqdm

from conf.gpt_4_vision_config import Gpt4VisionConfig
from conf.data_config import DataConfig
from data_enums.clevr_math_labels_enum import ClevrMathLabelsEnum
from gpt_clients.gpt4_vision_client import Gpt4VisionClient
from utils.logger import init_logger
from one_step_gpt import OneStepGPT


class OneStepGPTCot(OneStepGPT):
    """
    This class is used to solve the questions from CLEVR-math in a one-step process.
    It also uses a chain of thought prompting.
    """
    def __init__(self, data_config: DataConfig, gpt_client: Gpt4VisionClient, logger: Logger):
        super().__init__(data_config=data_config, gpt_client=gpt_client, logger=logger)
        self.cot_subtraction_image: str = data_config.cot_subtraction_image
        self.cot_addition_image: str = data_config.cot_addition_image
        self.cot_one_step_gpt_results_file: Path = data_config.one_step_gpt_cot_results_file
        self.one_step_gpt_results_file: Path = Path(__file__).parent.joinpath(
            "data",
            "validation_set_results",
            "one_step_gpt_results.json"
        )

    @property
    def prompt(self) -> str:
        prompt = (
            "Answer the following <question>, as demonstrated in the examples.\n"
            "Pay attention: make sure to conclude your answer with the following format: \n"
            "'My answer is: <numeric answer>'\n"
            "<question>: {question}"
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
            dataset = self.download_dataset(self.clevr_math_dataset_name)[ClevrMathLabelsEnum.CHOSEN_DATASET]

            for question_index in tqdm(one_step_gpt_results.keys()):
                question_index = int(question_index)
                question_data = dataset[question_index]
                template = question_data[ClevrMathLabelsEnum.TEMPLATE]
                question_result = self.get_question_result(question_data=question_data)
                results[question_index] = question_result
                # update counters
                self.questions_counter[template] += 1

        except Exception as e:
            self.logger.exception(f"Error while solving questions: {e}")
            raise e
        finally:
            return results

    def get_model_response(self, image_path: str, prompt: str) -> str:
        """
        This function is adding the chain of thought prompt to the model's request.
        """
        encoded_subtraction_image = self.gpt_client.encode_image(self.cot_subtraction_image)
        encoded_addition_image = self.gpt_client.encode_image(self.cot_addition_image)
        chain_of_thought_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are asked to answer a mathematical <question> about an image."
                                "Before answering the question, here are examples of question-answer pairs."
                                "Use this approach when answering the <question>.\n"
                                "Example 1: Subtract all brown things. Subtract all brown cylinders. "
                                "How many objects are left?"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encoded_subtraction_image}",
                    },
                ],

            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "a. There are 2 brown things in the image: 1 big brown metal cylinder and "
                                "1 big brown metal sphere.\n"
                                "b.There is 1 brown cylinder in the image: 1 big brown metal cylinder.\n"
                                "c. There are 6 objects in the image in total: 1 big brown metal cylinder, "
                                "1 big green metal cylinder, 1 gray small rubber cube, 1 cyan small rubber cube, "
                                "1 big blue rubber cube, 1 big brown metal sphere.\n"
                                "d. After subtracting the 2 brown things we are left with 4 objects in the image. "
                                "We don't need to subtract the brown cylinder again as we already subtracted"
                                "it as part of 'brown things'.\n "
                                "My answer is: 4."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Another example: Add two cyan blocks. How many cyan objects are there?"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encoded_addition_image}",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "a. There is 1 cyan object in the image: 1 large cyan metal cylinder.\n"
                                "b. After adding 2 cyan blocks, there are 3 cyan objects in the image.\n"
                                "My answer is: 3."
                    }
                ]
            }
        ]
        return self.gpt_client.get_vision_model_response(
            image_path=image_path,
            prompt=prompt,
            chain_of_thought_messages=chain_of_thought_messages
        )


if __name__ == "__main__":
    logger = init_logger(file_name="one_step_gpt.log")

    gpt_config = Gpt4VisionConfig()
    gpt_vision_client = Gpt4VisionClient(config=gpt_config, logger=logger)

    config = DataConfig()
    one_step_gpt_cot = OneStepGPTCot(data_config=config, gpt_client=gpt_vision_client, logger=logger)
    answers = one_step_gpt_cot.solve_questions()

    one_step_gpt_cot.save_json_file(file_path=one_step_gpt_cot.cot_one_step_gpt_results_file, data=answers)
    logger.info(f"Finished solving questions.")
    logger.info(f"Number of correct answers: {one_step_gpt_cot.get_number_of_correct_answers(results=answers)}")

