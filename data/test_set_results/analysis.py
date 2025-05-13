import json

TWO_STEP = "two_step_gpt_results_vision.json"
ONE_STEP = "one_step_gpt_results.json"
ONE_STEP_COT = "one_step_gpt_cot_results.json"

def load_json_file(file_path: str) -> dict:
    """
    Load a json file and return its content.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def load_sampled_keys_list(sampled_questions: str) -> set[str]:
    """
    Read the sampled keys from the file: Each line contains a key (int).
    Save the keys in a set and return it.
    """
    sampled_keys = set()
    with open(sampled_questions, "r") as f:
        for line in f:
            sampled_keys.add((line.strip()))
    return sampled_keys

def get_correct_detection_from_2step_results(sampled_keys: set[str], two_step_results:dict) -> set[int]:
    """
    Get the correct detection from the two step results.
    """
    correct_detection = set()
    for question_index, question_data in two_step_results.items():
        if question_index not in sampled_keys:
            continue
        if question_data["detection_validation"]:
            correct_detection.add(question_index)
    return correct_detection


def calculate_wrong_answers_out_of_correct_detections(correct_detection: set[int], results: dict[int, dict]) -> int:
    """
    Calculate the number of wrong answers out of the correct detections.
    """
    wrong_answers = 0
    for question_index in correct_detection:
        question_data = results[question_index]
        if not question_data["is_correct"]:
            wrong_answers += 1
    return wrong_answers


def main():
    sampled_keys = load_sampled_keys_list("sampled_keys_for_validation.txt")
    two_step_results = load_json_file(TWO_STEP)
    correct_detection = get_correct_detection_from_2step_results(sampled_keys, two_step_results=two_step_results)
    print("Correct detection: ", len(correct_detection))
    for file in [ONE_STEP, ONE_STEP_COT, TWO_STEP]:
        print(f"Approach: {file}")
        results = load_json_file(file)
        wrong_answers = calculate_wrong_answers_out_of_correct_detections(
            correct_detection=correct_detection,
            results=results
        )
        print("Wrong answers: ", wrong_answers)
        # print the wrong answers out of correct detection ratio with 2 decimal points
        print("Wrong answers out of correct detection ratio: ", round(wrong_answers / len(correct_detection), 2))


if __name__ == "__main__":
    main()