import asyncio
import os
import tempfile
from pathlib import Path
from threading import Lock

from google import genai
from google.genai import types # type: ignore
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.8-flash"
HISTORY_LIMIT = 10
CONTEXT_TURNS = 6

whisper_model = None
whisper_lock = Lock()


def crear_cliente_ia():
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta configurar GEMINI_API_KEY en PowerShell.")
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=45_000),
    )


def transcribir_audio(ruta_audio: str) -> str:
    """Transcribe una nota de voz en español usando Whisper local."""
    global whisper_model

    if WhisperModel is None:
        raise RuntimeError("Falta faster-whisper. Ejecuta: python -m pip install -r requirements.txt")

    if whisper_model is None:
        with whisper_lock:
            if whisper_model is None:
                # Base ofrece una carga más ligera que small en CPU.
                whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    segmentos, _ = whisper_model.transcribe(ruta_audio, language="es", vad_filter=True)
    return " ".join(segmento.text.strip() for segmento in segmentos).strip()


def generar_respuesta(pregunta: str, conversacion: list[dict]) -> str:
    """Genera una respuesta conversacional en español con Gemini."""
    cliente = crear_cliente_ia()
    instrucciones = (
        "Eres ChatMind IA, un asistente conversacional útil, amable y claro. "
        "Responde en español y contesta directamente la pregunta del usuario. "
        "Si hace falta información para responder, pide una aclaración. "
        "No inventes datos; reconoce cuando no sabes algo. Mantén la respuesta breve, "
        "a menos que el usuario pida una explicación detallada."
    )
    contexto = []
    for turno in conversacion[-CONTEXT_TURNS:]:
        contexto.append(f"Usuario: {turno['pregunta']}")
        contexto.append(f"ChatMind IA: {turno['respuesta']}")

    prompt = instrucciones
    if contexto:
        prompt += "\n\nConversación reciente:\n" + "\n".join(contexto)
    prompt += f"\n\nPregunta actual del usuario: {pregunta}"

    configuracion = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        max_output_tokens=500,
    )

    resultado = None
    for modelo in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            resultado = cliente.models.generate_content(
                model=modelo,
                contents=prompt,
                config=configuracion,
            )
            break
        except Exception as error:
            mensaje_error = str(error).upper()
            es_saturacion = "503" in mensaje_error or "UNAVAILABLE" in mensaje_error
            if not es_saturacion or modelo == GEMINI_FALLBACK_MODEL:
                raise
            print(f"{modelo} está ocupado; probando el modelo alternativo...")

    if resultado is None:
        return "No pude generar una respuesta en este momento. Intenta de nuevo en unos segundos."
    respuesta = (resultado.text or "").strip()
    return respuesta or "No pude generar una respuesta. ¿Puedes formular la pregunta de otra manera?"


def guardar_interaccion(context: ContextTypes.DEFAULT_TYPE, pregunta: str, respuesta: str):
    historial = context.user_data.setdefault("historial_chatmind", [])
    historial.append({"pregunta": pregunta[:500], "respuesta": respuesta[:1500]})
    del historial[:-HISTORY_LIMIT]


async def responder_pregunta(update: Update, context: ContextTypes.DEFAULT_TYPE, pregunta: str):
    mensaje = update.effective_message
    pregunta = pregunta.strip()
    if not pregunta:
        await mensaje.reply_text("No alcancé a reconocer una pregunta. Intenta escribirla o grabarla otra vez.")
        return

    progreso = await mensaje.reply_text("Estoy pensando en una respuesta…")
    conversacion = context.user_data.setdefault("conversacion_chatmind", [])
    try:
        print("Generando respuesta con Gemini...")
        respuesta = await asyncio.to_thread(generar_respuesta, pregunta, conversacion)
        conversacion.append({"pregunta": pregunta[:500], "respuesta": respuesta[:1500]})
        del conversacion[:-CONTEXT_TURNS]
        guardar_interaccion(context, pregunta, respuesta)
        await progreso.edit_text(respuesta[:4000])
        print("Respuesta enviada.")
    except Exception as error:
        print(f"Error al generar respuesta: {error}")
        await progreso.edit_text(
            "No pude obtener una respuesta de la IA. Revisa la conexión y la configuración de la clave de Gemini."
        )


async def responder_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await responder_pregunta(update, context, update.effective_message.text or "")


async def responder_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje = update.effective_message
    archivo_temporal = None
    try:
        await mensaje.reply_text("Recibí tu audio. Estoy transcribiendo la pregunta…")
        archivo_telegram = await (mensaje.voice or mensaje.audio).get_file()
        extension = ".ogg" if mensaje.voice else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporal:
            archivo_temporal = temporal.name

        await archivo_telegram.download_to_drive(custom_path=archivo_temporal)
        pregunta = await asyncio.to_thread(transcribir_audio, archivo_temporal)
        if not pregunta:
            await mensaje.reply_text("No reconocí palabras en el audio. Intenta grabarlo más cerca y con menos ruido.")
            return

        await mensaje.reply_text(f"Transcripción: {pregunta}")
        await responder_pregunta(update, context, pregunta)
    except Exception as error:
        print(f"Error al procesar audio: {error}")
        # Si la conexión con Telegram falló, puede fallar también este aviso.
        # Evitamos que el error termine como una excepción sin controlar.
        try:
            await mensaje.reply_text(
                "No pude procesar el audio por un problema de conexión con Telegram. "
                "Espera unos segundos y vuelve a enviarlo."
            )
        except Exception as error_aviso:
            print(f"No se pudo enviar el aviso del audio: {error_aviso}")
    finally:
        if archivo_temporal and os.path.exists(archivo_temporal):
            Path(archivo_temporal).unlink(missing_ok=True)


async def mostrar_historial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje = update.effective_message
    historial = context.user_data.get("historial_chatmind", [])
    if not historial:
        await mensaje.reply_text("Aún no tienes preguntas guardadas. Envíame una pregunta por texto o audio.")
        return

    lineas = ["Historial reciente de preguntas y respuestas:"]
    for numero, turno in enumerate(reversed(historial[-5:]), start=1):
        pregunta = turno["pregunta"].replace("\n", " ")[:180]
        respuesta = turno["respuesta"].replace("\n", " ")[:280]
        lineas.append(f"\n{numero}. Tú: {pregunta}\nChatMind IA: {respuesta}")
    await mensaje.reply_text("\n".join(lineas)[:4000])


async def nueva_conversacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["conversacion_chatmind"] = []
    await update.effective_message.reply_text(
        "Listo. Empezamos una conversación nueva. El historial anterior sigue disponible con /historial."
    )


def teclado_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Hacer una pregunta", callback_data="preguntar")],
        [InlineKeyboardButton("Ver historial", callback_data="historial")],
        [InlineKeyboardButton("Nueva conversación", callback_data="nueva")],
        [InlineKeyboardButton("Cómo funciona", callback_data="ayuda")],
    ])


async def mostrar_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "ChatMind IA\n\nPuedes escribirme una pregunta o enviarla como nota de voz.",
        reply_markup=teclado_menu(),
    )


async def mostrar_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Envíame una pregunta por texto o como nota de voz y te responderé. "
        "El bot conserva los últimos seis intercambios como contexto. "
        "Usa /historial para consultar los últimos cinco y /nuevo para empezar desde cero."
    )


async def atender_boton(update: Update, context: ContextTypes.DEFAULT_TYPE):
    consulta = update.callback_query
    await consulta.answer()
    if consulta.data == "preguntar":
        await consulta.message.reply_text("Escribe tu pregunta o graba una nota de voz.")
    elif consulta.data == "historial":
        await mostrar_historial(update, context)
    elif consulta.data == "nueva":
        await nueva_conversacion(update, context)
    elif consulta.data == "ayuda":
        await mostrar_ayuda(update, context)


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Falta configurar TELEGRAM_TOKEN en PowerShell.")
    if not GEMINI_API_KEY:
        raise SystemExit("Falta configurar GEMINI_API_KEY en PowerShell.")

    # La red estaba agotando el timeout predeterminado de 5 segundos al responder
    # al audio. Damos más margen a las solicitudes normales del Bot API.
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler("start", mostrar_menu))
    app.add_handler(CommandHandler("menu", mostrar_menu))
    app.add_handler(CommandHandler("ayuda", mostrar_ayuda))
    app.add_handler(CommandHandler("historial", mostrar_historial))
    app.add_handler(CommandHandler("nuevo", nueva_conversacion))
    app.add_handler(CallbackQueryHandler(atender_boton))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_texto))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, responder_audio))

    print("ChatMind IA iniciado: preguntas por texto y audio, contexto e historial activos.")
    app.run_polling()


if __name__ == "__main__":
    main()
