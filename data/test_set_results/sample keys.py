import json
import random


def sample_keys_from_json(json_file, sample_size=20):
    try:
        # Load JSON data
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Identify unique templates and organize keys by template
        templates = {}
        for key, value in data.items():
            template = value.get('template', 'unknown')
            if template not in templates:
                templates[template] = []
            templates[template].append(key)

        # Sample keys for each template
        sampled_keys = {}
        for template, keys in templates.items():
            if len(keys) > sample_size:
                sampled_keys[template] = random.sample(keys, sample_size)
            else:
                sampled_keys[template] = keys

        # Write sampled keys to a file
        with open('sampled_keys.txt', 'w') as output_file:
            for template, keys in sampled_keys.items():
                for key in keys:
                    output_file.write(f"{key}\n")

        print("Sampled keys successfully saved to 'sampled_keys.txt'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    json_file = 'data/test_set_results/two_step_gpt_results_vision.json'
    sample_keys_from_json(json_file)
