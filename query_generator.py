# Contains the logic for generating text prompts using an LLM.
# This is the first step of the DASH-LLM pipeline.

import torch
from typing import List

# two-step refinement process described in the DASH paper's appendix.

import torch
from typing import List


class DASH_LLM_QueryGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda"  # LLM is loaded on GPU with device_map="auto"

    def _get_initial_prompt(self, target_object: str) -> List[dict]:
        """
        Constructs the detailed initial system prompt from Figure 7 in the paper's appendix.
        """
        # This prompt is a direct transcription of the one in the paper.
        content = (
            """As an AI language model assistant, your task is to provide descriptive captions for images showing
spurious features.

 A spurious feature is a visual element that frequently co-occurs with a given object in images and may
cause AI models to incorrectly recognize the object, even when it is not present.

Task Overview:

You will be given:
- An object.

Your job is to:

1. Think of potential spurious features: Identify objects, scenes, or elements that frequently co-occur
with the given object in images. These should not include any parts or components of the object
itself.

2. Generate 50 unique and diverse prompts describing images that contain only these spurious features,
without including the object itself or any of its parts.

Important Guidelines:

- Do Not Mention the Object Name or Any Part of It: Avoid any direct or indirect references to the
object name. If the object name is a composite or compound word, do not include any part of the
object name in the prompts. For example, if the object is "firetruck," do not use "fire" or "truck"
in the prompts.

- Do Not Mention Parts of the Object: Do not include any parts or components of the object in the
prompts. For example, if the object is "mountainbike," do not use "handlebar," "gear shift," or "
saddle" in the prompts.

- Do Not Include the Object Name in Written Text: Do not create prompts that refer to written text
containing the object name or any part of it. For example, avoid descriptions like "a sign that
says ’hummingbird’."

- Focus on Spurious Features: Use features that are likely correlated with the object due to frequent
co-occurrence in images.

- Combining Elements: You may combine elements if they logically make sense to appear together in one
image. Do not combine elements unlikely to co-occur.

- Ensure Diversity: Each prompt should be unique and cover different aspects of the spurious features.

- Avoid Repetition: Do not repeat prompts or make minor variations of the same prompt.

- Style and Detail: Write clear, creative, and descriptive prompts. Keep each prompt concise.

- Language and Grammar: Use proper grammar and spelling.

- Content Restrictions: Do not include offensive, sensitive, or inappropriate content.

- Avoid Bias: Ensure prompts are inclusive and free from cultural, gender, or racial bias.

- Verification: Before submitting, review the prompts to ensure they comply with all guidelines.

Examples:

For the object "hummingbird":

- Correct Prompts:
- "Close-up of a bird feeder hanging in a lush garden."
- "A garden filled with vibrant red flowers."
- "Green foliage glistening after a rainfall."
- "A bird feeder surrounded by blooming plants."
- "Red tubular flowers swaying in the breeze."

- Incorrect Prompts (Do Not Use):
- "A hummingbird hovering near a flower."
- "Close-up of a hummingbird’s wings in motion."
- "A small bird with iridescent feathers perched on a branch."
- "A sign with the word ’hummingbird’ in a botanical garden."

For the object "firetruck":

- Correct Prompts:
- "A fire station with bright red doors."
- "Close-up of a spinning emergency siren light."
- "Firefighters conducting a training drill."
- "A tall ladder reaching up the side of a building."
- "Protective gear hanging neatly in a station locker room."

- Incorrect Prompts (Do Not Use):
- "A bright red firetruck parked on the street."
- "Children waving at a passing firetruck."
- "A sign that reads ’Fire Station No. 1’."
- "A red truck with emergency equipment."
- Using the words "fire" or "truck" in the prompts.

For the object "mountainbike":

- Correct Prompts:
- "A winding trail cutting through a dense forest."
- "A helmet resting on a tree stump beside a path."
- "Sunlight filtering through trees along a forest trail."
- "A backpack leaning against a wooden signpost on a hillside."
- "A group of friends hiking through mountainous terrain."

- Incorrect Prompts (Do Not Use):
- "A mountainbike leaning against a tree."
- "Close-up of a mountainbike’s gears."
- "A cyclist adjusting the saddle of a mountainbike."
- "A sign that says ’Mountainbike Trail Ahead’."
- Using the words "mountain" or "bike" in the prompts.
- Mentioning parts like "handlebar," "gear shift," or "saddle."

Formatting Instructions:

- Start each prompt on a new line, numbered sequentially from 1 to 50.

- The format should be:

1: <prompt_1>
2: <prompt_2>
3: <prompt_3>
...
50: <prompt_50>

User Input Format:

The user will provide the object in the following format:

object: <object name>

Your Response:

- Return exactly 50 prompts per user request.

- Ensure that the last line of your response starts with:

50: <prompt_50>

- Under no circumstances should you include any content in your response other than the 50 prompts. Do
not include explanations, apologies, or any additional text.

Summary:

- Do not mention the object name or any part of it. If the object name is a composite or compound word,
do not include any part of it in the prompts.

- Do not mention parts or components of the object.

- Do not create prompts that refer to written text containing the object name or any part of it.

- Focus on spurious features that frequently co-occur with the object.

- You may combine elements if they logically co-occur in an image.

- Ensure diversity and uniqueness in the prompts.

- Use proper language and avoid any inappropriate content.

- Review all prompts for compliance before submitting.

- Under no circumstances should you include any content in your response other than the 50 prompts. Do
not include explanations, apologies, or any additional text.

Remember, the goal is to create prompts that could lead an AI model to falsely recognize the object due
to the presence of spurious features, even though the object itself is not present in the images."""
        )
        return [
            {"role": "system", "content": content},
            {"role": "user", "content": f"object: {target_object}"}
        ]

    def _get_refinement_prompt(self, previous_response: str) -> List[dict]:
        """
        Constructs the follow-up prompt from Figure 8 in the paper's appendix,
        asking the LLM to correct its own previous output.
        """
        content = (
"""Please review the list of prompts you previously generated and check for any mistakes or deviations
from the guidelines. Identify any prompts that do not fully comply with the instructions. Then,
generate a new list of 50 prompts that strictly adhere to all the guidelines provided.

Important Guidelines:

- Do not mention the object name or any part of it. If the object name is a composite or compound word,
do not include any part of the object name in the prompts.
- Do not mention parts or components of the object.
- Do not create prompts that refer to written text containing the object name or any part of it.
- Focus on spurious features that frequently co-occur with the object.
- You may combine elements if they logically co-occur in an image.
- Ensure diversity and uniqueness in the prompts.
- Use proper language and avoid any inappropriate content.
- Review all prompts for compliance before submitting.
- Under no circumstances should you include any content in your response other than the 50 prompts. Do
not include explanations, apologies, or any additional text.

Formatting Instructions:

- Start each prompt on a new line, numbered sequentially from 1 to 50.
- The format should be:

1: <prompt_1>
2: <prompt_2>
3: <prompt_3>
...
50: <prompt_50>

- Ensure that the last line of your response starts with:

50: <prompt_50>

Remember, your goal is to create prompts that could lead an AI model to falsely recognize the object
due to the presence of spurious features, even though the object itself is not present in the
images.

Now, generate the corrected list of 50 prompts."""
        )
        return [
            {"role": "system", "content": content},
            {"role": "user", "content": f"Here is the list to review and correct:\n\n{previous_response}"},
            {"role": "assistant", "content": "Now, generate the corrected list of 50 prompts."}
        ]

    def _call_llm(self, messages: List[dict]) -> str:
        """Helper function to call the LLM and get a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40096,  # Increased token limit for the detailed prompt
            num_return_sequences=1,
            do_sample=True,
            temperature=0.6,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    def generate(self, target_object: str, num_queries: int = 50) -> List[str]:
        """
        Generates and refines a list of text queries using the paper's two-step process.
        """
        # --- Step 1: Initial Generation ---
        print("Generating initial list of prompts...")
        initial_messages = self._get_initial_prompt(target_object)
        initial_response = self._call_llm(initial_messages)

        # --- Step 2: Refinement ---
        print("Refining the list of prompts...")
        refinement_messages = self._get_refinement_prompt(initial_response)
        final_response = self._call_llm(refinement_messages)

        # --- Post-processing ---
        queries = []
        for line in final_response.split('\n'):
            line = line.strip()
            if ':' in line:
                line = line.split(':', 1)[1].strip()
            elif '.' in line:
                line = line.split('.', 1)[1].strip()

            line = line.replace('"', '').replace("'", "").strip()
            if line and len(line) > 5:  # Stricter quality filter
                queries.append(line)

        # Final check to ensure the object name is not included
        queries = [q for q in queries if target_object.lower() not in q.lower()]

        unique_queries = sorted(list(set(queries)))
        return unique_queries[:num_queries]
