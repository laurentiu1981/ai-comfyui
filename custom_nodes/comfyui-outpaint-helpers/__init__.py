"""Outpaint helpers for this repo's scripts/outpaint.sh.

QwenImage21FunInpaint: drive the inpaint channels of the Qwen Image 2.1 Fun
ControlNet Union (keep mask + masked-image latents) without a structural control
image. ComfyUI's own QwenImageDiffsynthControlnet node requires `image` (the
structural control) and does not expose `inpaint_image`; feeding the padded
canvas as the control image makes the model paint the padding as a flat colour.
"""

import comfy.ldm.qwen_image21.model
from comfy_extras.nodes_model_patch import QwenImage21FunControlPatch


class QwenImage21FunInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_patch": ("MODEL_PATCH",),
                "vae": ("VAE",),
                "inpaint_image": ("IMAGE", {"tooltip": "Canvas with the known pixels; masked pixels are replaced by mid-gray before encoding."}),
                "mask": ("MASK", {"tooltip": "1 (white) = regenerate, 0 = keep."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "control_image": ("IMAGE", {"tooltip": "Optional structural control (canny/depth/...); zeros when not given."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patch/qwen"
    DESCRIPTION = "Qwen Image 2.1 Fun ControlNet Union in inpaint mode: regenerate the masked area of inpaint_image."

    def apply(self, model, model_patch, vae, inpaint_image, mask, strength=1.0, control_image=None, start_percent=0.0, end_percent=1.0):
        if not isinstance(model_patch.model, comfy.ldm.qwen_image21.model.QwenImage21FunControl):
            raise ValueError("model_patch is not a Qwen Image 2.1 Fun ControlNet")
        if strength == 0:
            return (model,)
        model_patched = model.clone()
        model_sampling = model.get_model_object("model_sampling")
        sigma_range = (float(model_sampling.percent_to_sigma(start_percent)), float(model_sampling.percent_to_sigma(end_percent)))
        if control_image is not None:
            control_image = control_image[:, :, :, :3]
        QwenImage21FunControlPatch(model_patch, vae, control_image, strength,
                                   inpaint_image=inpaint_image[:, :, :, :3], mask=mask,
                                   sigma_range=sigma_range).register(model_patched)
        return (model_patched,)


NODE_CLASS_MAPPINGS = {"QwenImage21FunInpaint": QwenImage21FunInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21FunInpaint": "Qwen Image 2.1 Fun ControlNet Inpaint"}
