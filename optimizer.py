# optimizer.py
# Implements the core image generation and optimization logic for DASH-OPT.

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import config


class DASH_OPT_Optimizer:
    def __init__(self, vlm, vlm_processor, object_detector, od_processor, device):
        self.vlm = vlm
        self.vlm_processor = vlm_processor
        self.object_detector = object_detector
        self.od_processor = od_processor
        self.device = device
        self._load_diffusion_model()

    def _load_diffusion_model(self):
        """
        Loads the distilled Stable Diffusion XL model and LCM scheduler as per the paper's appendix.
        """
        print("Loading Distilled SDXL model for DASH-OPT...")

        # As per paper Appendix C, using a distilled version of SDXL.
        # We use a 1-step LCM LoRA model for speed and efficiency.
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.HF_HOME,
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.HF_HOME,
        ).to(self.device)

        # Set scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        print("Distilled SDXL model loaded.")

    @torch.no_grad()
    def _get_detector_confidence(self, image: Image.Image, object_label: str) -> torch.Tensor:
        """Gets the raw max confidence from the object detector."""
        texts = [[f"a photo of a {object_label}"]]
        inputs = self.od_processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.object_detector(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.od_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                  threshold=0.0)  # No threshold to get all scores

        scores = results[0]['scores']
        if len(scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return scores.max()

    def _get_vlm_yes_probability(self, image: Image.Image, object_label: str) -> torch.Tensor:
        """
        Gets the VLM's probability for the token 'Yes'.
        This requires getting the logits for the next token prediction.
        """
        prompt = f"Can you see a {object_label} in this image? Please answer only with yes or no."

        # Manually encode to get both image and text inputs
        inputs = self.vlm_processor(text=prompt, images=image, return_tensors="pt").to(self.device,
                                                                                       dtype=torch.bfloat16)

        # Get the logits from the VLM
        with torch.no_grad():
            outputs = self.vlm(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for the last token

        # Find the token ID for "Yes"
        yes_token_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]

        # Calculate probability using softmax
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[yes_token_id]

    def generate_image(self, text_prompt: str, object_label: str):
        """
        The main optimization loop to generate a single hallucination-inducing image.
        """
        # Encode the initial prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(text_prompt)
        prompt_embeds.requires_grad_()

        # Paper Appendix C: Optimize the conditioning variables
        # We optimize the prompt embeddings. A random latent is drawn each step.
        optimizer = torch.optim.Adam([prompt_embeds], lr=0.1)

        best_loss = float('inf')
        best_image = None

        pbar = tqdm(range(config.DASH_OPT_STEPS), desc=f"Optimizing for '{object_label}'")
        for step in pbar:
            optimizer.zero_grad()

            # Generate image from current embeddings
            latents = torch.randn((1, 4, 128, 128), device=self.device, dtype=torch.float16)
            image = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                latents=latents,
                num_inference_steps=4,  # Fast generation with LCM
                guidance_scale=1.0,
                output_type="pil"
            ).images[0]

            # 1. VLM Loss
            prob_yes = self._get_vlm_yes_probability(image, object_label)
            loss_vlm = -torch.log(prob_yes + 1e-9)  # Add epsilon for stability

            # 2. Detector Loss
            detector_confidence = self._get_detector_confidence(image, object_label)
            # Threshold confidence as per Appendix C
            detector_confidence_thresholded = torch.clamp(detector_confidence - 0.05, min=0.0)
            loss_det = -torch.log(1 - detector_confidence_thresholded + 1e-9)

            # 3. Total Loss
            total_loss = loss_vlm + config.DASH_OPT_LAMBDA * loss_det

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{total_loss.item():.2f}", "P(Yes)": f"{prob_yes.item():.2f}",
                              "Det_Conf": f"{detector_confidence.item():.2f}"})

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_image = image

        return best_image