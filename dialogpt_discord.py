import discord
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch._dynamo
import os
from dotenv import load_dotenv

load_dotenv()
torch._dynamo.config.suppress_errors = True

tokenizer = AutoTokenizer.from_pretrained("pineappleSoup/DialoGPT-medium-707")
model = AutoModelForCausalLM.from_pretrained("pineappleSoup/DialoGPT-medium-707")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

def interact_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_length=200,
        num_return_sequences=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print("------")

@bot.command(name="chat")
async def chat(ctx, *, prompt: str):
    response = interact_with_model(prompt)
    await ctx.send(f"{response}")

@bot.command(name="stop")
async def stop(ctx):
    await ctx.send("Goodbye!")
    await bot.close()

# Run the bot
if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    if not DISCORD_TOKEN:
        raise ValueError("No Discord token found in .env file. Please add DISCORD_TOKEN=your_token to .env.")
    bot.run(DISCORD_TOKEN)