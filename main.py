# AI CheckMate — Telegram bot for contract risk analysis
# ------------------------------------------------------
# Replace placeholders below with your own API keys or set ENV variables:
#   export TELEGRAM_BOT_TOKEN="<your‑telegram-token>"
#   export OPENAI_API_KEY="<your-openai-api-key>"
# ------------------------------------------------------

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.enums.parse_mode import ParseMode
from aiogram.types import FSInputFile

import openai
import pdfplumber
import pytesseract
from PIL import Image
import docx2txt

# --- Configuration ---------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "<TELEGRAM_BOT_TOKEN>")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<OPENAI_API_KEY>")

# OpenAI set‑up
openai.api_key = OPENAI_API_KEY
_MODEL = "gpt-4o-mini"  # or the model you are entitled to use
_MAX_TOKENS_RESPONSE = 1200

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

# --------------------  Utils ------------------------------------------------

def extract_text_from_pdf(path: Path) -> str:
    """Extracts text from a PDF using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def extract_text_from_image(path: Path) -> str:
    """OCR text from an image (jpg, png, etc.)."""
    img = Image.open(path)
    # you can tweak pytesseract config if needed
    return pytesseract.image_to_string(img, lang="rus+eng")


def extract_text_from_docx(path: Path) -> str:
    return docx2txt.process(str(path))


def guess_document_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    return ext  # 'pdf', 'docx', 'jpg', etc.


async def analyze_document_text(text: str) -> str:
    """Call the LLM to analyse contract text and return nicely formatted result."""
    system_prompt = (
        "Ты — виртуальный юрист‑ассистент. Проанализируй текст договора, найди потенциальные "
        "риски и двусмысленные формулировки. Для каждого риска сформируй: (1) короткое название, "
        "(2) цитату из договора, (3) почему опасно простыми словами, (4) совет, как исправить." 
        "Заверши списком коротких пунктов‑чек‑листа, что проверить перед подписанием."
    )

    response = await openai.AsyncOpenAI().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:60_000]},  # safety clip to ~60k chars
        ],
        max_tokens=_MAX_TOKENS_RESPONSE,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# --------------------  Handlers --------------------------------------------

@dp.message(commands={"start", "help"})
async def cmd_start(message: types.Message):
    await message.reply(
        "👋 <b>AI CheckMate</b> — проверяю договоры на риски.\n"\
        "<i>Просто отправь PDF, DOCX или фото — и я выделю опасные места и объясню человеческим языком.</i>"
    )


@dp.message(content_types={types.ContentType.DOCUMENT, types.ContentType.PHOTO})
async def handle_document(message: types.Message):
    await message.reply("📥 Получил файл, анализирую… ⏳")

    # 1. Download file to a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        if message.content_type == types.ContentType.DOCUMENT:
            file_id = message.document.file_id
            filename = message.document.file_name or "document"
        else:  # PHOTO — take the highest resolution
            file_id = message.photo[-1].file_id
            filename = f"photo_{file_id}.jpg"

        file_path = Path(tmpdir) / filename
        await bot.download(file_id, destination=file_path)

        # 2. Extract text
        doc_type = guess_document_type(filename)
        try:
            if doc_type == "pdf":
                text = extract_text_from_pdf(file_path)
            elif doc_type in {"doc", "docx"}:
                text = extract_text_from_docx(file_path)
            elif doc_type in {"jpg", "jpeg", "png", "heic", "webp"}:
                text = extract_text_from_image(file_path)
            else:
                await message.reply("❌ Не могу обработать этот тип файла.")
                return
        except Exception as e:
            logger.exception("Extract error: %s", e)
            await message.reply("⚠️ Не удалось извлечь текст из файла. Попробуйте другой формат.")
            return

        if len(text.strip()) < 200:
            await message.reply("⚠️ Документ почти пустой или текст не распознан.")
            return

        # 3. Analyze via LLM
        try:
            analysis_result = await analyze_document_text(text)
        except Exception as e:
            logger.exception("LLM error: %s", e)
            await message.reply("⚠️ Не удалось проанализировать документ. Попробуйте позже.")
            return

        # 4. Send back result
        await message.reply(analysis_result, disable_web_page_preview=True)


# --------------------  Entry point -----------------------------------------

async def main() -> None:
    logger.info("Starting AI‑CheckMate bot …")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")

# --------------------  requirements.txt  -----------------------------------


# --------------------  Dockerfile  -----------------------------------------
"""
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "ai_checkmate_bot.py"]
"""
