# import torch
# from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
# from diffusers.utils import export_to_gif

# adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
# pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
# pipe.set_adapters(["lcm-lora"], [0.8])

# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

# output = pipe(
#     prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
#     negative_prompt="bad quality, worse quality, low resolution",
#     num_frames=16,
#     guidance_scale=2.0,
#     num_inference_steps=6,
#     generator=torch.Generator("cpu").manual_seed(0),
# )
# frames = output.frames[0]
# export_to_gif(frames, "animatelcm.gif")

# import torch
# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# from diffusers.utils import export_to_video

# # load pipeline
# pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32, variant="fp16")
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# # generate
# prompt = "Spiderman is surfing. Darth Vader is also surfing and following Spiderman"
# video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames

# # convert to video
# video_path = export_to_video(video_frames)

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "a brown man in casuals sitting on chair"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")

