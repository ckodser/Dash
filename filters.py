# Contains the filter classes used in the Exploration phase.
# - ObjectDetectorFilter: Checks if an object is physically present.
# - VLMFilter: Checks if the VLM *thinks* an object is present.

import torch
from PIL import Image

import config


class ObjectDetectorFilter:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def is_object_present(self, image: Image.Image, object_label: str) -> bool:
        """
        Uses OWLv2 to detect if the object is in the image.

        Returns:
            True if the object is detected with confidence > threshold, False otherwise.
        """
        texts = [[f"a photo of a {object_label}"]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Target image sizes (as specified by the processor)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        # Post-process the outputs
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                               threshold=config.OD_CONFIDENCE_THRESHOLD)

        # If any box is found for our query, the scores list will not be empty
        scores = results[0]['scores']
        return len(scores) > 0


class VLMFilter:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def is_object_present(self, image: Image.Image, object_label: str) -> bool:
        """
        Asks the VLM if it sees the object in the image.

        Returns:
            True if the VLM's response contains "yes", False otherwise.
        """
        # The paper uses a specific prompt format.
        prompt = f"Can you see a {object_label} in this image? Please answer only with yes or no."

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Handle potential dtype issues
        if self.device == "cuda":
            inputs = {k: v.to(torch.bfloat16) for k, v in inputs.items()}

        generate_ids = self.model.generate(**inputs, max_new_tokens=10)

        response = \
        self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # The response is often in the format: `prompt\n\nANSWER`. We extract the answer.
        answer = response.split(prompt)[-1].strip().lower()

        return "yes" in answer
