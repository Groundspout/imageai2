
import discord
from discord.ext import commands
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,UniPCMultistepScheduler
import torch
import asyncio
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from discord import app_commands
executor = ThreadPoolExecutor()
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())





model_id = "stabilityai/stable-diffusion-2-base"
queue = []
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")



semaphore = threading.Semaphore(value=1)

from collections import deque
from typing import Deque, Tuple

queue: Deque[Tuple[int, discord.Message]] = deque()

async def update_queue_messages():
    for i, (_, message) in enumerate(queue):
        await message.edit(content=f"Generating image... (position in queue: {i + 1})")

def generate_image(prompt, id):
    semaphore.acquire()
    try:
        image = pipe(prompt,num_inference_steps=50,guidance_scale=8,negative_prompt="ugly").images[0]
        prompt = prompt.replace("\\","")
        path = f"{random.randint(1,100000000000000000000000000000000000000000000000)}.png"
        image.save(path)
        file = discord.File(path)
    finally:
        semaphore.release()
        queue.popleft()
        asyncio.run_coroutine_threadsafe(update_queue_messages(), bot.loop)
    return (file,path)




@bot.event
async def on_ready():
    await bot.tree.sync()

@bot.tree.command(name="ai", description="Generate an image using AI, describe the image with up to 100 characters.")
async def generate(interaction:discord.Interaction, prompt: str=None):
    if not prompt or len(prompt)>100:
        await interaction.response.send_message("Please write a description of the image up to 100 characters.")
    position = len(queue) + 1
    await interaction.response.send_message(f"Generating image... (position in queue: {position})")
    response = await interaction.original_response()
    queue.append((interaction.user.id, response))
    asyncio.run_coroutine_threadsafe(update_queue_messages(), bot.loop)
    value = await bot.loop.run_in_executor(executor, generate_image, prompt, interaction.user.id)
    image = value[0]
    if image:
        await interaction.followup.send(content=prompt, file=image)
        os.remove(value[1])
    else:
        await interaction.followup.send("there was an error, try again later...")
    await response.edit(content="Finished generating")



bot.run(
  'MTExMzEwNDIyNzIyMTM4MTE1MA.GoPxZb.d0w449b8n3DMxasP1-HHiJo7nRCT1YnHQ20hYY')
