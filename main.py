import os
import sys
import asyncio
import logging
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import find_dotenv, load_dotenv

from src.llm_working import get_answer


# Load envs
load_dotenv(find_dotenv())
TOKEN = getenv("TOKEN")
print(TOKEN)

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {message.from_user.first_name}! I am an AI bot that can answer questions about Operating Systems. Ask me questions about Operating Systems and I will answer them.")

@dp.message()
async def handle_question(message: Message):
    result = get_answer(message.text)
    #result = chain.invoke(message.text)
    await message.reply(result)


async def main():
    print("in main")
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
