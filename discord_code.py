import discord
from discord.ext import commands

intents = discord.Intents.default()

bot = commands.Bot(command_prefix='!',intents=intents)

@bot.command()
async def hello(ctx):
    await ctx.send('Hello World!')

bot.run('MTA4MTQ3MzAzNTYwMjc3NjIwNg.GHoVAo.DX8y2vqvowvleHfHvq1K8VesCMvkOpp7fIznuE')
