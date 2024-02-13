from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path

load_dotenv()


@dataclass
class DataConfig:
    clevr_val_scenes: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "CLEVR_val_scenes.json"),
        metadata={"help": "The name of the CLEVR validation scenes file."},
    )
    one_step_gpt_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "one_step_gpt_results.json"),
        metadata={"help": "The name of the file where all the one step gpt results are saved."},
    )
    one_step_gpt_cot_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "one_step_gpt_cot_results.json"),
        metadata={"help": "The name of the file where all the one step gpt cot results are saved."},
    )
    objects_parsing_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "objects_parsing_results.json"),
        metadata={"help": "The name of the file where all the objects parsing results are saved."},
    )
    object_counting_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "object_counting_results.json"),
        metadata={"help": "The name of the file where all the object counting results are saved."},
    )
    object_counting_validation_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "object_counting_validation.json"),
        metadata={"help": "The name of the file where all the object counting validation results are saved."},
    )
    two_step_gpt_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "two_step_gpt_results.json"),
        metadata={"help": "The name of the file where all the two step gpt results are saved."},
    )
    two_step_gpt_vision_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath("data", "two_step_gpt_results_vision.json"),
        metadata={"help": "The name of the file where all the two step gpt results are saved."},
    )
    oracle_one_step_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath(
            "data", "validation_set_results", "oracle_one_step_results.json"
        ),
        metadata={"help": "The name of the file where all the oracle one step results are saved."},
    )
    oracle_parsing_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath(
            "data", "validation_set_results", "oracle_parsing_results.json"
        ),
        metadata={"help": "The name of the file where all the oracle parsing results are saved."},
    )
    oracle_two_step_results_file: Path = field(
        default=Path(__file__).parent.parent.joinpath(
            "data", "validation_set_results", "oracle_two_step_results.json"
        ),
        metadata={"help": "The name of the file where all the oracle two step results are saved."},
    )
    number_of_questions_to_solve: int = field(default=400)
    clevr_math_dataset_name: str = field(default="dali-does/clevr-math")
    cot_subtraction_image: str = "data/CLEVR_train_000006.png"
    cot_addition_image: str = "data/CLEVR_train_000000.png"
