# TwoStageGPT

This is the code used for the paper "Decomposed Prompting for Vision and Language Arithmetic Reasoning".
This code is using [CLEVR Math](https://arxiv.org/pdf/2208.05358.pdf) dataset to test GPT-4 model with vision capabilities in solving vision and
language word math problems.

**To run the code:**

1. Create your own Azure open AI Service resource as
   described [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal) -
   make sure you have both GPT-4 vision model and GPT-4 language model deployed as this code uses both of the models.
2. Create an .env file in the root directory of the project and add the following variables:
```
GPT4_LANG_KEY=<YOUR_GPT4_LANGUAGE_MODEL_KEY>
GPT4_LANG_ENDPOINT=<YOUR_GPT4_LANGUAGE_MODEL_ENDPOINT>
GPT4_LANG_DEPLOYMENT_NAME=<YOUR_GPT4_LANGUAGE_MODEL_DEPLOYMENT_NAME>

GPT4_VISION_KEY=<YOUR_GPT4_VISION_MODEL_KEY>
GPT4_VISION_ENDPOINT=<YOUR_GPT4_VISION_MODEL_ENDPOINT>
GPT4_VISION_DEPLOYMENT_NAME=<YOUR_GPT4_VISION_MODEL_DEPLOYMENT_NAME>
```
3. Install the required packages by running `pip install -r requirements.txt`


## Experiments
### One-stage approach
This approach uses the GPT-4 model with vision capabilities to solve the math problems directly: the model 
is given the math problem and the image and is expected to output the answer.
There are three different experiments in this approach:
1. one_step_gpt.py: This experiment uses the GPT-4 model with vision capabilities to solve the math problems directly.
2. one_step_gpt_CoT.py: This experiment uses the GPT-4 model with vision capabilities to solve the math problems directly, 
   using a chain of thought prompting.
3. oracle_one_step.py: In this experiment, we solve the questions from CLEVR-math in a one-step approach.
    However, in this case, in addition to the image and the question, the model also receives a full 
    description of all the objects in the image (taken from CLEVR dataset annotations).

### Two-stage approach
This approach uses the GPT-4 model with vision capabilities to solve the math problems in using decomposed prompting:
the GPT-4 language model is provided with the question, 
and is required to indentify which objects are relevant to the question.   
Then, the GPT-4 vision model is provided with the relevant objects and the image, 
and it is required to enumerate and count these objects. 
Finally, the GPT-4 vision model is provided with these descriptions, the question and the image, and it is required to
solve the question.
There are two different experiments in this approach:
1. two_step_gpt_vision.py: This experiment uses the GPT-4 model with vision capabilities to solve the math problems 
using decomposed prompting.
2. oracle_two_step.py: In this experiment, we solve the questions from CLEVR-math in a two-stage approach. 
However, in this case, in addition to the image and the question, the model also receives a full 
description of the relevant objects in the image (taken from CLEVR dataset annotations). This experiment failed because 
the parser that was used to extract the relevant objects from the CLEVR dataset annotations was not able to extract
them as expected. 

## Run the Experiments
### One-stage approach
1. to run the one_step_gpt.py experiment, run `python one_step_gpt.py`
2. to run the one_step_gpt_CoT.py experiment, run `python one_step_gpt_CoT.py`
3. to run the oracle_one_step.py experiment, run `python oracle_one_step.py`

### Two-stage approach
1. to run the two_step_gpt_vision.py experiment, run the files in the following order:
    1. `python objects_parser.py`
    2. `python objects_counter.py`
    3. `python two_step_gpt_vision.py`
2. to run the oracle_two_step.py experiment, run `python oracle_two_step.py`, run the files in the following order:
    1. `python oracle_parser.py`
    2. `python oracle_two_step.py`

