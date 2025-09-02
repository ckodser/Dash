# optimizer.py
# Implements the core image generation and optimization logic for DASH-OPT.

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

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

        unet = UNet2DConditionModel.from_pretrained(
            config.SDXL_UNET_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.HF_HOME,
        ).to("cuda:2")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            config.SDXL_BASE_MODEL_ID,
            unet=unet,
            # config.SDXL_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.HF_HOME,
        ).to("cuda:2")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.set_timesteps(timesteps=[config.DASH_OPT_START_TIMESTEP])
        # self.pipe.scheduler.set_timesteps(num_inference_steps=config.DASH_OPT_INFERENCE_STEPS)
        print("Distilled SDXL model loaded.")

    def _get_detector_confidence(self, image_tensor: torch.Tensor, object_label: str) -> torch.Tensor:
        """Gets the raw max confidence from the object detector from a tensor."""
        texts = [[f"a photo of a {object_label}"]]
        
        # We need to manually normalize the tensor for the OD processor
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.object_detector.device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.object_detector.device)
        # The input tensor for OD needs to be float32
        normalized_tensor = ((image_tensor.to(torch.float32).to(self.object_detector.device) / 255.0) - image_mean) / image_std
        
        inputs = self.od_processor(text=texts, images=normalized_tensor, return_tensors="pt").to(self.object_detector.device)
        
        outputs = self.object_detector(**inputs)

        h, w = image_tensor.shape[-2:]
        target_sizes = torch.Tensor([[h, w]]).to(self.object_detector.device)
        
        results = self.od_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.0)
        
        scores = results[0]['scores']
        if len(scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return scores.max()


    def _get_vlm_yes_probability(self, image_tensor: torch.Tensor, object_label: str) -> torch.Tensor:
        """
        Gets the VLM's probability for the token 'Yes' from a tensor.
        """
        prompt = f"<image>\nCan you see a {object_label} in this image? Please answer only with yes or no."
        
        # The VLM expects a specific input format, handle this carefully
        # We pass the raw tensor for pixel_values
        print("MIN:", torch.min(image_tensor), "MAX:", torch.max(image_tensor))
        inputs = self.vlm_processor(text=prompt, images=image_tensor, return_tensors="pt")
        for k in inputs.keys():
            print("INPUT #", k, inputs[k].shape)
        inputs = {k: v.to(self.vlm.device) for k, v in inputs.items()}

        # No need for torch.no_grad here as we want the gradients for VLM loss
        outputs = self.vlm(**inputs)
        logits = outputs.logits[0, -1, :]
        
        yes_token_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]
        
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[yes_token_id]

    def _chi_square_loss(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Chi-Square regularization loss for the latent vector.
        Encourages the L2 norm of the latent to stay close to sqrt of its dimension.
        """
        n = float(latents.numel())
        expected_norm = torch.sqrt(n)
        latent_norm = torch.norm(latents)
        return torch.pow(latent_norm - expected_norm, 2)

    def generate_image(self, text_prompt: str, object_label: str):
        """
        The main optimization loop to generate a single hallucination-inducing image.
        """
        scaler = GradScaler()
        
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                text_prompt)

        prompt_embeds_leaf = prompt_embeds.clone().to(torch.float32).requires_grad_(True)
        pooled_prompt_embeds_leaf = pooled_prompt_embeds.clone().to(torch.float32).requires_grad_(True)
        print("prompt_embeds_leaf", prompt_embeds_leaf)
        print("pooled_prompt_embeds_leaf", pooled_prompt_embeds_leaf)
        latents_leaf = torch.randn(
            (1, config.DASH_OPT_LATENT_CHANNELS, config.DASH_OPT_LATENT_HEIGHT, config.DASH_OPT_LATENT_WIDTH),
            device=self.pipe.device,
            dtype=torch.float32
        ).requires_grad_(True)
        print("latents_leaf", latents_leaf)
       # --- 2. Setup Optimizer and Scheduler ---
        # Create parameter groups for differential learning rates
        param_groups = [
            {'params': [prompt_embeds_leaf, pooled_prompt_embeds_leaf], 'lr': config.DASH_OPT_LR},
            {'params': [latents_leaf], 'lr': config.DASH_OPT_LR * config.DASH_OPT_LATENT_LR_FACTOR}
        ]
        optimizer = torch.optim.Adam(param_groups)
        
        # Setup linear learning rate warmup scheduler
        lr_lambda = lambda step: min(1.0, (step + 1) / config.DASH_OPT_WARMUP_STEPS)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Define the image metadata for SDXL.
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        
        # Concatenate the metadata and convert to a tensor.
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32, device=self.pipe.device)

    
        best_loss = float('inf')
        best_image = None

        pbar = tqdm(range(config.DASH_OPT_STEPS), desc=f"Optimizing for '{object_label}'")
        for step in pbar:
            optimizer.zero_grad()

            with autocast():
                # --- 3. Diffusion Forward Pass ---
                # Scale the latents by the scheduler's initial sigma
                scaled_latents = latents_leaf * self.pipe.scheduler.init_noise_sigma
                
                # We pass the specific start timestep to the LCM scheduler
                timesteps = torch.tensor([config.DASH_OPT_START_TIMESTEP], device=self.pipe.device)
    
                # Generate the image tensor for loss calculation
                noise_pred = self.pipe.unet(
                    scaled_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_leaf,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds_leaf, "time_ids": add_time_ids}
                ).sample
                print("noise_pred", noise_pred)
                # Get the image from the predicted noise
                image_tensor = self.pipe.scheduler.step(noise_pred, timesteps[0], scaled_latents, return_dict=False)[0]
                image_tensor_denoised = self.pipe.vae.decode(image_tensor / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                print("image_tensor", image_tensor)
                print("image_tensor_denoised", image_tensor_denoised)
                # --- 4. Calculate Losses ---
                # VLM Loss
                print("TORCH MIN VAE OUTPUT: ", torch.min(image_tensor_denoised), " MAX:", torch.max(image_tensor_denoised))
                vlm_input_tensor = (image_tensor_denoised / 2 + 0.5).clamp(0, 1) # VAE output is [-1, 1], convert to [0, 1]
                prob_yes = self._get_vlm_yes_probability(vlm_input_tensor, object_label)
                loss_vlm = -torch.log(prob_yes + 1e-9)
    
                # Detector Loss
                detector_confidence = self._get_detector_confidence((vlm_input_tensor * 255).detach(), object_label)
                detector_confidence_thresholded = torch.clamp(detector_confidence - config.DASH_OPT_DETECTOR_THRESHOLD, min=0.0)
                loss_det = -torch.log(1 - detector_confidence_thresholded + 1e-9)
                
                # Chi-Square Regularization Loss for the latent
                loss_chi_sq = self._chi_square_loss(latents_leaf)
    
                # Total Loss
                total_loss = loss_vlm.to(loss_chi_sq.device) + loss_det.to(loss_chi_sq.device) + config.DASH_OPT_CHI_SQUARE_REG_LAMBDA * loss_chi_sq

            scaler.scale(total_loss).backward()
            s
            # --- 5. Backpropagation and Optimizer Step ---
            
            # Gradient Clipping
            all_params = [p for group in param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_params, config.DASH_OPT_GRAD_CLIP_NORM)

            with torch.no_grad():
                grad_norm_prompt = torch.norm(prompt_embeds_leaf.grad).item() if prompt_embeds_leaf.grad is not None else 0
                grad_norm_pooled = torch.norm(pooled_prompt_embeds_leaf.grad).item() if pooled_prompt_embeds_leaf.grad is not None else 0
                grad_norm_latent = torch.norm(latents.grad).item() if latents.grad is not None else 0

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.2f}", 
                "LR": f"{scheduler.get_last_lr()[0]:.4f}",
                "P(Yes)": f"{prob_yes.item():.2f}",
                "GradNorm(Prompt)": f"{grad_norm_prompt:.4f}",
                "GradNorm(Pooled)": f"{grad_norm_pooled:.4f}",
                "GradNormLatent": f"{grad_norm_latent:.4f}"
            })

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                # Convert the best tensor to a PIL image for saving
                pil_image = self.pipe.image_processor.postprocess(image_tensor_denoised.detach(), output_type="pil")[0]
                best_image = pil_image

        return best_image