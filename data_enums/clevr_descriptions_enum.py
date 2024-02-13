from enum import Enum


class ClevrDescriptionsEnum(str, Enum):
    """"
    The labels for the clevr descriptions dataset:

    """
    SCENES = "scenes"
    OBJECTS = "objects"
    IMAGE_ID = "image_filename"
