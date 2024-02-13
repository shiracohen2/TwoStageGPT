from enum import Enum


class ImageDataEnum(str, Enum):
    QUESTION_INDEX = "question_index"
    IMAGE_PATH = "image_path"
    IMAGE_ID = "image_id"
    QUESTION = "question"
    TEMPLATE = "template"
    GPT_RESPONSE = "gpt_response"
    NUMERICAL_RESULT = "numerical_result"
    IS_CORRECT = "is_correct"
    LABEL = "label"
    PARSING_RESULT = "parsing_result"
    COUNTING_RESULT = "counting_result"
    IS_VALID = "is_valid"
    VALIDATION_EXPLANATION = "explanation"

