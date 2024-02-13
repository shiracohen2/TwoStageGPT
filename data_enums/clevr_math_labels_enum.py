from enum import Enum


class ClevrMathLabelsEnum(str, Enum):
    """"
    The labels for the clevr math dataset:
    1. Template - the template is the question type (adversarial, subtraction-multihop, addition, subtraction, etc.)
    2. Question - the question text
    3. Image - the image data - this one is an object
    4. Label - the correct answer for the question
    5. Id - this is the image name in the dataset, for e.g.: CLEVR_train_000003.png

    """
    TEMPLATE = "template"
    QUESTION = "question"
    IMAGE = "image"
    ID = "id"
    LABEL = "label"
    # The following enums represents the chosen split: train, validation or test
    CHOSEN_DATASET = "test"
