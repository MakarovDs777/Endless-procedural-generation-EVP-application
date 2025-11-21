import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import struct
import threading
import random
import numpy as np
import sounddevice as sd
from pydub import AudioSegment  # остаётся для экспортов/создания mp3 при необходимости

# Параметры по умолчанию
DEFAULT_SAMPLE_RATE = 44100  # Гц
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 1  # bytes (8-bit) — для режима bytes

# Глобальные переменные для управления циклом случайного воспроизведения
random_loop_thread = None
random_loop_stop_event = None

def parse_number_array(text):
    if not text:
        raise ValueError("Пустой ввод.")
    cleaned = text.strip()
    lines = cleaned.splitlines()
    meta = None
    if lines and lines[0].strip().startswith('#META'):
        header = lines[0].strip()[5:].strip()
        meta = {}
        for part in header.split():
            if '=' in part:
                k, v = part.split('=', 1)
                if re.fullmatch(r"\d+", v):
                    meta[k] = int(v)
                else:
                    meta[k] = v
        nums_text = '\n'.join(lines[1:])
    else:
        nums_text = cleaned

    nums = re.findall(r"-?\d+", nums_text)
    if not nums:
        raise ValueError("Не найдено чисел в вводе или файле.")
    ints = [int(s) for s in nums]
    for n in ints:
        if n < 0 or n > 255:
            raise ValueError(f"Число вне диапазона 0..255: {n}")
    return ints, meta

def _make_audio_from_bytes(raw_bytes, frame_rate, channels, sample_width):
    """Создаёт AudioSegment из сырых байтов (для экспортов), без добавления тишины."""
    return AudioSegment(data=bytes(raw_bytes), sample_width=sample_width, frame_rate=frame_rate, channels=channels)

def _bytes_to_float32(raw_bytes, sample_width, channels):
    """
    Преобразует сырые PCM-байты в float32 в диапазоне [-1.0, 1.0] для sounddevice.
    Поддерживаем типичные 8-bit unsigned и 16-bit signed.
    """
    if len(raw_bytes) == 0:
        return np.zeros((0,), dtype=np.float32)

    if sample_width == 1:
        # 8-bit PCM обычно unsigned: 0..255 -> -1..1
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0
    else:
        # fallback: попробуем интерпретировать как int16
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0

    if channels > 1:
        try:
            arr = arr.reshape(-1, channels)
        except Exception:
            # если reshape не удался, попытаемся добавить ось и дублировать
            arr = arr.reshape(-1, 1)
            if arr.shape[1] != channels:
                # продублируем канал в случае несовпадения
                arr = np.tile(arr, (1, channels))
    return arr.astype(np.float32)

def _loop_play_random_from_file(numbers, meta, stop_event):
    """Открывает sounddevice OutputStream один раз и в цикле пишет в него сгенерированные случайные непрерывные фрагменты (минимизируем паузы)."""
    if not numbers:
        return

    # Параметры аудио
    frame_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    sample_width = DEFAULT_SAMPLE_WIDTH
    if meta:
        try:
            if 'sample_rate' in meta:
                frame_rate = int(meta['sample_rate'])
            if 'channels' in meta:
                channels = int(meta['channels'])
            if 'sample_width' in meta:
                sample_width = int(meta['sample_width'])
        except Exception:
            pass

    # Параметры генерации
    min_chunk_ms = 100   # минимальный кусок в ms
    max_chunk_ms = 2000  # максимальный кусок в ms
    play_buffer_ms = 5000  # длина буфера для одной записи в поток (ms)

    bytes_per_sec = frame_rate * channels * sample_width

    byte_array = bytes(numbers)
    nbytes = len(byte_array)
    if nbytes == 0:
        return

    # Открываем единый поток для вывода (dtype float32)
    try:
        with sd.OutputStream(samplerate=frame_rate, channels=channels, dtype='float32', latency='low') as stream:
            while not stop_event.is_set():
                # формируем один большой непрерывный байтовый буфер
                buffer_bytes = bytearray()
                buffer_ms_acc = 0
                while buffer_ms_acc < play_buffer_ms and not stop_event.is_set():
                    chunk_ms = random.randint(min_chunk_ms, max_chunk_ms)
                    needed_bytes = max(1, int(bytes_per_sec * (chunk_ms / 1000.0)))
                    if needed_bytes <= nbytes:
                        start = random.randint(0, nbytes - needed_bytes)
                        sel = byte_array[start:start + needed_bytes]
                    else:
                        reps = (needed_bytes + nbytes - 1) // nbytes
                        sel = (byte_array * reps)[:needed_bytes]
                    buffer_bytes.extend(sel)
                    # приближённо пересчитываем длину в ms
                    buffer_ms_acc = int((len(buffer_bytes) / bytes_per_sec) * 1000)

                if len(buffer_bytes) == 0:
                    continue

                # Конвертируем в float32 и пишем в поток — запись происходит непрерывно в открытый поток
                audio_np = _bytes_to_float32(buffer_bytes, sample_width, channels)
                try:
                    stream.write(audio_np)
                except Exception:
                    # если запись упала — даём короткий таймаут и пробуем снова
                    if stop_event.wait(0.05):
                        break
                    continue
    except Exception as e:
        # При ошибке открытия аудиоустройства — покажем пользователю (GUI-поток не блокируем здесь)
        try:
            messagebox.showerror("Ошибка аудио", f"Не удалось открыть аудиопоток: {e}")
        except Exception:
            pass
        return

# --- GUI ---
root = tk.Tk()
root.title("Конвертер чисел <-> MP3/Audio — Random from file (gapless via sounddevice)")
root.geometry("980x620")

info = ("Кнопка 'Случайный звук из файла' попросит выбрать текстовый файл с числами (0..255).\n"
        "Файл будет интерпретирован как режим bytes (как если бы в первой строке было '#META mode=bytes ...'),\n"
        "и будет воспроизводиться непрерывный поток случайных участков без пауз (используется sounddevice).")

tk.Label(root, text=info, justify=tk.LEFT).pack(pady=12)

text_input = tk.Text(root, height=12, width=120)
text_input.pack(pady=6)
text_input.insert("1.0", "Пример содержимого файла:\n#META mode=bytes sample_rate=44100 channels=1 sample_width=1\n0 127 255 34 200 ...")

button_frame = tk.Frame(root)
button_frame.pack(pady=12)

def on_create_from_text():
    raw_text = text_input.get("1.0", tk.END).strip()
    try:
        numbers, meta = parse_number_array(raw_text)
    except ValueError as e:
        messagebox.showerror("Ошибка ввода", str(e))
        return
    try:
        frame_rate = meta.get('sample_rate', DEFAULT_SAMPLE_RATE) if meta else DEFAULT_SAMPLE_RATE
        channels = meta.get('channels', DEFAULT_CHANNELS) if meta else DEFAULT_CHANNELS
        sample_width = meta.get('sample_width', DEFAULT_SAMPLE_WIDTH) if meta else DEFAULT_SAMPLE_WIDTH
        raw = bytes(numbers)
        audio = AudioSegment(data=raw, sample_width=sample_width, frame_rate=frame_rate, channels=channels)
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_path = os.path.join(desktop, "from_text_bytes.mp3")
        audio.export(out_path, format='mp3')
    except Exception as e:
        messagebox.showerror("Ошибка при создании MP3", str(e))
        return
    messagebox.showinfo("Готово", f"MP3 сохранён:\n{out_path}")

def on_random_sound_from_file_toggle():
    global random_loop_thread, random_loop_stop_event

    # Если цикл уже запущен — остановим
    if random_loop_thread and random_loop_thread.is_alive():
        random_loop_stop_event.set()
        random_loop_thread.join(timeout=2.0)
        random_loop_thread = None
        random_loop_stop_event = None
        messagebox.showinfo("Стоп", "Бесконечный цикл случайных звуков остановлен.")
        random_file_btn.config(text="Случайный звук из файла (Старт)")
        return

    # Выбрать файл
    file_path = filedialog.askopenfilename(title="Выберите текстовый файл с числами",
                                           filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
        return

    try:
        numbers, meta = parse_number_array(data)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return

    # Готово — запускаем цикл из данных файла
    stop_event = threading.Event()
    random_loop_stop_event = stop_event
    thread = threading.Thread(target=_loop_play_random_from_file, args=(numbers, meta, stop_event), daemon=True)
    random_loop_thread = thread
    thread.start()
    messagebox.showinfo("Старт", "Запущен бесконечный цикл случайных звуков из выбранного файла (нажатие кнопки остановит).")
    random_file_btn.config(text="Случайный звук из файла (Стоп)")

# Кнопка "Создать MP3 из поля ввода" удалена — остальной интерфейс не изменён

random_file_btn = tk.Button(button_frame, text="Случайный звук из файла (Старт)",
                            command=on_random_sound_from_file_toggle, width=40)
random_file_btn.grid(row=0, column=0, padx=8, pady=4)

quit_btn = tk.Button(root, text="Выйти", command=root.quit, width=12)
quit_btn.pack(pady=12)

root.mainloop()

