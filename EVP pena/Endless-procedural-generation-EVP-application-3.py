import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import threading
import random
import time
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# Параметры по умолчанию
DEFAULT_SAMPLE_RATE = 44100  # Гц
DEFAULT_CHANNELS = 2
DEFAULT_SAMPLE_WIDTH = 2  # bytes (16-bit)

# Глобальные переменные для управления циклах
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


def _bytes_to_float32(raw_bytes, sample_width, channels):
    if len(raw_bytes) == 0:
        return np.zeros((0,), dtype=np.float32)

    frame_size = sample_width * max(1, channels)
    rem = len(raw_bytes) % frame_size
    if rem != 0:
        need = frame_size - rem
        raw_bytes = raw_bytes + (raw_bytes[:need] if len(raw_bytes) >= need else (raw_bytes * ((need // len(raw_bytes)) + 1))[:need])

    if sample_width == 1:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        # интерпретируем байты как signed 16-bit little-endian
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0
    else:
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0

    if channels > 1:
        try:
            arr = arr.reshape(-1, channels)
        except Exception:
            frames = arr.size // channels
            arr = arr[:frames * channels].reshape(frames, channels)
    return arr.astype(np.float32)


def _loop_play_random_from_file(numbers, meta, stop_event):
    # Воспроизведение из файла: выбираются случайные ФРЕЙМЫ (по frame_size байт)
    if not numbers:
        return

    frame_rate = meta.get('sample_rate', DEFAULT_SAMPLE_RATE) if meta else DEFAULT_SAMPLE_RATE
    channels = meta.get('channels', DEFAULT_CHANNELS) if meta else DEFAULT_CHANNELS
    sample_width = meta.get('sample_width', DEFAULT_SAMPLE_WIDTH) if meta else DEFAULT_SAMPLE_WIDTH

    min_chunk_ms = 100
    max_chunk_ms = 2000
    play_buffer_ms = 5000

    bytes_per_sec = frame_rate * channels * sample_width

    byte_array = bytes(numbers)
    nbytes = len(byte_array)
    if nbytes == 0:
        return

    frame_size = max(1, sample_width * channels)
    rem = nbytes % frame_size
    if rem != 0:
        need = frame_size - rem
        byte_array = byte_array + (byte_array[:need] if len(byte_array) >= need else (byte_array * ((need // len(byte_array)) + 1))[:need])
        nbytes = len(byte_array)

    try:
        with sd.OutputStream(samplerate=frame_rate, channels=channels, dtype='float32', latency='low') as stream:
            while not stop_event.is_set():
                buffer_bytes = bytearray()
                buffer_ms_acc = 0
                while buffer_ms_acc < play_buffer_ms and not stop_event.is_set():
                    chunk_ms = random.randint(min_chunk_ms, max_chunk_ms)
                    needed_bytes = max(1, int(bytes_per_sec * (chunk_ms / 1000.0)))
                    # выравниваем по frame_size
                    if needed_bytes % frame_size != 0:
                        needed_bytes += (frame_size - (needed_bytes % frame_size))

                    # Вместо CONTIGUOUS selection — формируем выборку из случайных фреймов
                    needed_frames = max(1, needed_bytes // frame_size)
                    sel = bytearray()
                    total_frames = nbytes // frame_size
                    if total_frames >= 1:
                        # выбираем нужное количество фреймов случайно (с повторениями)
                        for _ in range(needed_frames):
                            fi = random.randint(0, total_frames - 1)
                            start = fi * frame_size
                            sel.extend(byte_array[start:start + frame_size])
                    else:
                        # fallback: просто растянем весь массив
                        reps = (needed_bytes + nbytes - 1) // nbytes
                        sel = (byte_array * reps)[:needed_bytes]

                    buffer_bytes.extend(sel)
                    buffer_ms_acc = int((len(buffer_bytes) / bytes_per_sec) * 1000)

                if len(buffer_bytes) == 0:
                    continue

                audio_np = _bytes_to_float32(buffer_bytes, sample_width, channels)
                try:
                    stream.write(audio_np)
                except Exception:
                    if stop_event.wait(0.05):
                        break
                    continue
    except Exception as e:
        try:
            messagebox.showerror("Ошибка аудио", f"Не удалось открыть аудиопоток: {e}")
        except Exception:
            pass
        return


def _start_white_noise_stream(meta, stop_event, highlight_callback=None, volume=0.3, highlight_interval=0.08, allowed_numbers=None):
    """
    Генерирует белый шум. Если allowed_numbers передан (список 0..255), то вместо uniform генерируем байты
    из этого набора и конвертируем их в float через _bytes_to_float32 (с учётом sample_width).
    """
    sample_rate = int(meta.get('sample_rate', DEFAULT_SAMPLE_RATE)) if meta else DEFAULT_SAMPLE_RATE
    channels = int(meta.get('channels', DEFAULT_CHANNELS)) if meta else DEFAULT_CHANNELS
    sample_width = int(meta.get('sample_width', DEFAULT_SAMPLE_WIDTH)) if meta else DEFAULT_SAMPLE_WIDTH

    sample_rate = max(8000, sample_rate)
    blocksize = 1024

    def callback(outdata, frames, time_info, status):
        if status:
            pass
        if stop_event.is_set():
            raise sd.CallbackStop()

        if allowed_numbers:
            # генерируем байты из allowed_numbers
            nbytes = frames * channels * sample_width
            chosen = random.choices(allowed_numbers, k=nbytes)
            raw = bytes(chosen)
            audio_np = _bytes_to_float32(raw, sample_width, channels)
            # умножаем на volume
            try:
                outdata[:] = (audio_np * volume)
            except Exception:
                # fallback на uniform
                if channels == 1:
                    noise = np.random.uniform(-1.0, 1.0, frames).astype(np.float32)
                    out = (noise * volume).reshape(frames, 1)
                else:
                    noise = np.random.uniform(-1.0, 1.0, (frames, channels)).astype(np.float32)
                    out = noise * volume
                outdata[:] = out
        else:
            if channels == 1:
                noise = np.random.uniform(-1.0, 1.0, frames).astype(np.float32)
                out = (noise * volume).reshape(frames, 1)
            else:
                noise = np.random.uniform(-1.0, 1.0, (frames, channels)).astype(np.float32)
                out = noise * volume
            outdata[:] = out

        callback._counter += frames
        if callback._counter >= sample_rate * highlight_interval:
            callback._counter = 0
            num = random.randint(0, 255) if not allowed_numbers else random.choice(allowed_numbers)
            if highlight_callback:
                try:
                    highlight_callback(num)
                except Exception:
                    pass

    callback._counter = 0

    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype='float32', blocksize=blocksize, callback=callback)
        stream.start()
        while not stop_event.wait(0.1):
            pass
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
    except Exception as e:
        try:
            messagebox.showerror("Ошибка аудио", f"Не удалось открыть аудиопоток (EGF white-noise): {e}")
        except Exception:
            pass
        return


def _start_white_noise_from_bytes(numbers, meta, stop_event, highlight_callback=None, volume=0.35, highlight_interval=0.08, allowed_numbers=None):
    """
    Воспроизводит непрерывный поток. Если allowed_numbers указан — генерируем байты только из этого набора,
    иначе используем random-фреймы из переданных numbers/byte_array.
    """
    # Если нет данных — ничего не делаем
    if not numbers and not allowed_numbers:
        return
    # Берём параметры из метаданных
    base_rate = int(meta.get('sample_rate', DEFAULT_SAMPLE_RATE)) if meta else DEFAULT_SAMPLE_RATE
    channels = int(meta.get('channels', DEFAULT_CHANNELS)) if meta else DEFAULT_CHANNELS
    sample_width = int(meta.get('sample_width', DEFAULT_SAMPLE_WIDTH)) if meta else DEFAULT_SAMPLE_WIDTH

    frame_rate = max(8000, base_rate)
    frame_size = max(1, sample_width * channels)

    byte_array = bytes(numbers) if numbers else b''
    nbytes = len(byte_array)

    if nbytes > 0:
        rem = nbytes % frame_size
        if rem != 0:
            need = frame_size - rem
            byte_array = byte_array + (byte_array[:need] if len(byte_array) >= need else (byte_array * ((need // len(byte_array)) + 1))[:need])
            nbytes = len(byte_array)

    bytes_per_sec = frame_rate * channels * sample_width

    min_chunk_ms = 100
    max_chunk_ms = 2000
    play_buffer_ms = 5000

    try:
        with sd.OutputStream(samplerate=frame_rate, channels=channels, dtype='float32', latency='low') as stream:
            while not stop_event.is_set():
                buffer_bytes = bytearray()
                buffer_ms_acc = 0
                while buffer_ms_acc < play_buffer_ms and not stop_event.is_set():
                    chunk_ms = random.randint(min_chunk_ms, max_chunk_ms)
                    needed_bytes = max(1, int(bytes_per_sec * (chunk_ms / 1000.0)))
                    if needed_bytes % frame_size != 0:
                        needed_bytes += (frame_size - (needed_bytes % frame_size))

                    if allowed_numbers:
                        # генерируем данные строго из allowed_numbers
                        chosen = random.choices(allowed_numbers, k=needed_bytes)
                        sel = bytes(chosen)
                    else:
                        if needed_bytes <= nbytes and nbytes > 0:
                            max_start = nbytes - needed_bytes
                            if max_start <= 0:
                                start = 0
                            else:
                                steps = max_start // frame_size
                                start = random.randint(0, steps) * frame_size
                            sel = byte_array[start:start + needed_bytes]
                        else:
                            reps = (needed_bytes + nbytes - 1) // nbytes if nbytes > 0 else 1
                            sel = (byte_array * reps)[:needed_bytes]

                    buffer_bytes.extend(sel)
                    buffer_ms_acc = int((len(buffer_bytes) / bytes_per_sec) * 1000)

                if len(buffer_bytes) == 0:
                    continue

                audio_np = _bytes_to_float32(buffer_bytes, sample_width, channels)
                try:
                    stream.write(audio_np)
                except Exception:
                    if stop_event.wait(0.05):
                        break
                    continue

                if highlight_callback:
                    try:
                        buffer_seconds = len(buffer_bytes) / bytes_per_sec
                        hits = max(1, int(buffer_seconds / highlight_interval))
                        for _ in range(hits):
                            if stop_event.is_set():
                                break
                            num = random.randint(0, 255) if not allowed_numbers else random.choice(allowed_numbers)
                            try:
                                highlight_callback(num)
                            except Exception:
                                pass
                    except Exception:
                        pass
    except Exception as e:
        try:
            messagebox.showerror("Ошибка аудио", f"Не удалось открыть аудиопоток (EGF from bytes): {e}")
        except Exception:
            pass
        return


# --- GUI: комбинированное приложение (звуки + табло чисел 0..255) ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Конвертер чисел <-> MP3 + Табло 0..255 + EGF (white-noise)")
        self.geometry("1280x760")

        info = ("В этом окне можно: загрузить файл с числами (0..255) и запустить бесконечный\n"
                "циклический поток случайных фрагментов из файла (gapless через sounddevice),\n"
                "запустить EGF трансляцию — теперь это может быть непрерывный поток из фрагментов файла\n"
                "(если в поле указаны байты и #META mode=bytes),\n"
                "и подсветка на табло будет показывать случайные числа 0..255 с небольшой частотой.\n"
                "Дополнительно: можно задать Ограниченный EVP диапазон — тогда EGF будет генерировать аудио только из выбранных байтов.)")

        tk.Label(self, text=info, justify=tk.LEFT).pack(pady=8)

        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10)

        self.text_input = tk.Text(top_frame, height=6, width=120)
        self.text_input.pack(side=tk.LEFT, padx=(0,8), pady=6)
        self.text_input.insert("1.0", "Пример содержимого файла:\n#META mode=bytes sample_rate=44100 channels=2 sample_width=2\n0 127 255 34 200 ...")

        btns_frame = tk.Frame(top_frame)
        btns_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.random_file_btn = tk.Button(btns_frame, text="Случайный звук из файла (Старт)",
                                         command=self.on_random_sound_from_file_toggle, width=30)
        self.random_file_btn.pack(pady=4)

        self.load_file_btn = tk.Button(btns_frame, text="Загрузить файл в поле ввода...",
                                       command=self.load_file_to_input, width=30)
        self.load_file_btn.pack(pady=4)

        self.create_mp3_btn = tk.Button(btns_frame, text="Экспортировать из поля в MP3",
                                        command=self.on_create_from_text, width=30)
        self.create_mp3_btn.pack(pady=4)

        self.egf_btn = tk.Button(btns_frame, text="EGF трансляция (Старт)", command=self.on_egf_toggle, width=30)
        self.egf_btn.pack(pady=8)

        # Новая кнопка: Ограниченный EVP диапазон
        self.evp_allowed = None  # список разрешённых байтов (или None)
        self.evp_btn = tk.Button(btns_frame, text="Ограниченный EVP диапазон", command=self.on_evp_range_toggle, width=30)
        self.evp_btn.pack(pady=4)
        self.evp_label = tk.Label(btns_frame, text="EVP: нет ограничений", wraplength=220, justify=tk.LEFT)
        self.evp_label.pack(pady=(0,6))

        sep = tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN)
        sep.pack(fill=tk.X, padx=5, pady=8)

        board_frame = tk.Frame(self)
        board_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        controls_frame = tk.Frame(board_frame)
        controls_frame.pack(anchor='w')

        tk.Label(controls_frame, text="Строк (rows):").grid(row=0, column=0, sticky="w")
        self.rows_var = tk.IntVar(value=16)
        self.rows_spin = tk.Spinbox(controls_frame, from_=1, to=256, width=6, textvariable=self.rows_var)
        self.rows_spin.grid(row=0, column=1, padx=6)

        tk.Label(controls_frame, text="Столбцов (cols):").grid(row=0, column=2, sticky="w")
        self.cols_var = tk.IntVar(value=16)
        self.cols_spin = tk.Spinbox(controls_frame, from_=1, to=256, width=6, textvariable=self.cols_var)
        self.cols_spin.grid(row=0, column=3, padx=6)

        tk.Label(controls_frame, text="Интервал (сек):").grid(row=0, column=4, sticky="w")
        self.interval_var = tk.StringVar(value=str(0.6))
        self.interval_entry = tk.Entry(controls_frame, width=6, textvariable=self.interval_var)
        self.interval_entry.grid(row=0, column=5, padx=6)

        # Новое: регулировка скорости EGF (коэффициент). По умолчанию 0.5 (в 2 раза медленнее)
        tk.Label(controls_frame, text="Скорость EGF (коэффициент):").grid(row=0, column=6, sticky="w", padx=(12,0))
        self.egf_speed_var = tk.DoubleVar(value=0.5)
        self.egf_speed_entry = tk.Entry(controls_frame, width=6, textvariable=self.egf_speed_var)
        self.egf_speed_entry.grid(row=0, column=7, padx=6)
        tk.Label(controls_frame, text="(1.0=оригинал, 0.5=в 2 раза медленнее)").grid(row=0, column=8, sticky='w')

        self.order_var = tk.StringVar(value='sequential')
        tk.Radiobutton(controls_frame, text='Последовательный 0..255', variable=self.order_var, value='sequential').grid(row=1, column=0, columnspan=3, sticky='w', pady=(6,0))
        tk.Radiobutton(controls_frame, text='Перемешанный (shuffle)', variable=self.order_var, value='shuffle').grid(row=1, column=3, columnspan=3, sticky='w', pady=(6,0))

        self.start_btn = tk.Button(controls_frame, text="Старт табло", command=self.start_board)
        self.start_btn.grid(row=2, column=0, padx=6, pady=8)
        self.stop_btn = tk.Button(controls_frame, text="Стоп табло", command=self.stop_board, state=tk.DISABLED)
        self.stop_btn.grid(row=2, column=1, padx=6, pady=8)

        self.randomize_btn = tk.Button(controls_frame, text="Обновить сейчас", command=self.generate_board_once)
        self.randomize_btn.grid(row=2, column=2, padx=6, pady=8)

        self.save_board_btn = tk.Button(controls_frame, text="Сохранить табло на рабочий стол", command=self.save_board)
        self.save_board_btn.grid(row=2, column=3, padx=6, pady=8)

        self.board_text = tk.Text(board_frame, wrap=tk.NONE, font=("Consolas", 12))
        self.board_text.pack(fill=tk.BOTH, expand=True)
        self.board_text.configure(state=tk.DISABLED)

        self._board_running = False
        self._board_job = None
        self._board_numbers = list(range(256))

        self._egf_thread = None
        self._egf_stop_event = None
        self._egf_running = False

        self._init_board()
        self._render_board()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- file / sound helpers ---
    def load_file_to_input(self):
        fp = filedialog.askopenfilename(title="Выберите текстовый файл с числами",
                                        filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
        if not fp:
            return
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = f.read()
            self.text_input.delete('1.0', tk.END)
            self.text_input.insert('1.0', data)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

    def on_create_from_text(self):
        raw_text = self.text_input.get("1.0", tk.END).strip()
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

    def on_random_sound_from_file_toggle(self):
        global random_loop_thread, random_loop_stop_event

        if random_loop_thread and random_loop_thread.is_alive():
            random_loop_stop_event.set()
            random_loop_thread.join(timeout=2.0)
            random_loop_thread = None
            random_loop_stop_event = None
            messagebox.showinfo("Стоп", "Бесконечный цикл случайных звуков остановлен.")
            self.random_file_btn.config(text="Случайный звук из файла (Старт)")
            return

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

        stop_event = threading.Event()
        random_loop_stop_event = stop_event
        thread = threading.Thread(target=_loop_play_random_from_file, args=(numbers, meta, stop_event), daemon=True)
        random_loop_thread = thread
        thread.start()
        messagebox.showinfo("Старт", "Запущен бесконечный цикл случайных звуков из выбранного файла (нажатие кнопки остановит).")
        self.random_file_btn.config(text="Случайный звук из файла (Стоп)")

    # --- EGF: white-noise / from bytes ---
    def on_egf_toggle(self):
        if self._egf_running:
            # остановка
            self._egf_stop_event.set()
            if self._egf_thread:
                self._egf_thread.join(timeout=2.0)
            self._egf_thread = None
            self._egf_stop_event = None
            self._egf_running = False
            self.egf_btn.config(text="EGF трансляция (Старт)")
            return

        # старт: определяем коэффициент скорости
        try:
            coeff = float(self.egf_speed_var.get())
            if coeff <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Ошибка параметра", "Скорость EGF должна быть положительным числом. Используйте дробные значения, например 0.5 для замедления.")
            return

        # Попробуем взять содержимое поля ввода и, если там mode=bytes, использовать его как источник EGF
        raw_text = self.text_input.get("1.0", tk.END).strip()
        use_bytes_source = False
        numbers = None
        meta = None
        try:
            numbers, meta = parse_number_array(raw_text)
            if meta and str(meta.get('mode', '')).lower() == 'bytes':
                use_bytes_source = True
        except Exception:
            # не фатально — будем использовать белый шум
            use_bytes_source = False

        stop_event = threading.Event()
        self._egf_stop_event = stop_event

        def highlight_cb(num):
            try:
                self.after(0, lambda: self._highlight_number(num))
            except Exception:
                pass

        # передадим выбранные пользователем allowed numbers
        allowed = self.evp_allowed if hasattr(self, 'evp_allowed') else None

        if use_bytes_source and numbers:
            # используем метаданные, но применяем коэффициент скорости к sample_rate
            meta_play = dict(meta) if meta else {}
            base_sr = int(meta_play.get('sample_rate', DEFAULT_SAMPLE_RATE))
            meta_play['sample_rate'] = max(8000, int(base_sr * coeff))
            thread = threading.Thread(target=_start_white_noise_from_bytes, args=(numbers, meta_play, stop_event, highlight_cb, 0.35, 0.08, allowed), daemon=True)
        else:
            # fallback: обычный white-noise, скорость регулируем через изменение sample_rate
            meta_play = {'sample_rate': max(8000, int(DEFAULT_SAMPLE_RATE * coeff)), 'channels': DEFAULT_CHANNELS, 'sample_width': DEFAULT_SAMPLE_WIDTH}
            thread = threading.Thread(target=_start_white_noise_stream, args=(meta_play, stop_event, highlight_cb, 0.35, 0.08, allowed), daemon=True)

        self._egf_thread = thread
        self._egf_running = True
        thread.start()
        self.egf_btn.config(text="EGF трансляция (Стоп)")

    def on_evp_range_toggle(self):
        # Если уже установлено — очистим
        if self.evp_allowed:
            self.evp_allowed = None
            self.evp_label.config(text="EVP: нет ограничений")
            messagebox.showinfo("EVP", "Ограничение EVP удалено.")
            return

        file_path = filedialog.askopenfilename(title="Выберите текстовый файл с числами EVP",
                                               filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
            return

        nums = re.findall(r"\d+", data)
        if not nums:
            messagebox.showerror("Ошибка", "В файле не найдено чисел 0..255.")
            return
        ints = [int(s) for s in nums]
        cleaned = [n for n in ints if 0 <= n <= 255]
        if not cleaned:
            messagebox.showerror("Ошибка", "В файле не найдено допустимых чисел 0..255.")
            return
        # уникализируем и сохраним
        allowed = sorted(set(cleaned))
        self.evp_allowed = allowed
        preview = ','.join(str(x) for x in allowed[:20]) + (',...' if len(allowed) > 20 else '')
        self.evp_label.config(text=f"EVP: {len(allowed)} значений\n{preview}")
        messagebox.showinfo("EVP установлен", f"Разрешённые байты загружены: {len(allowed)} значений.")

    def _highlight_number(self, n):
        try:
            rows = max(1, int(self.rows_var.get()))
            cols = max(1, int(self.cols_var.get()))
        except Exception:
            return
        total = rows * cols
        try:
            idx = self._board_numbers.index(n)
        except ValueError:
            return
        if idx >= total:
            return
        r = idx // cols
        c = idx % cols
        # позиция символа: каждый элемент занимает 4 символа (" 12 ")
        start_char = c * 4
        line = r + 1
        start = f"{line}.{start_char}"
        end = f"{line}.{start_char + 3}"
        try:
            self.board_text.tag_remove("egf_highlight", "1.0", tk.END)
            self.board_text.tag_add("egf_highlight", start, end)
            self.board_text.tag_config("egf_highlight", background="yellow")
            self.board_text.see(start)
        except Exception:
            pass

    # --- board (табло 0..255) ---
    def _init_board(self):
        self._board_numbers = list(range(256))

    def _render_board(self):
        rows = max(1, int(self.rows_var.get()))
        cols = max(1, int(self.cols_var.get()))
        nums = self._board_numbers

        total = rows * cols
        if total <= 0:
            lines = [""]
        else:
            if len(nums) < total:
                nums2 = nums + [''] * (total - len(nums))
            else:
                nums2 = nums[:total]

            lines = []
            for r in range(rows):
                row_slice = nums2[r * cols:(r + 1) * cols]
                formatted = [f"{n:3}" if isinstance(n, int) else '   ' for n in row_slice]
                lines.append(" ".join(formatted))

        text_out = "\n".join(lines)
        self.board_text.configure(state=tk.NORMAL)
        self.board_text.delete("1.0", tk.END)
        self.board_text.insert(tk.END, text_out)
        self.board_text.configure(state=tk.DISABLED)

    def _update_board_once(self):
        if self.order_var.get() == 'shuffle':
            nums = list(range(256))
            random.shuffle(nums)
            self._board_numbers = nums
        else:
            self._board_numbers = list(range(256))
        self._render_board()

    def generate_board_once(self):
        try:
            self._update_board_once()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _board_tick(self):
        if not self._board_running:
            return
        try:
            self._update_board_once()
        except Exception:
            pass
        try:
            interval = float(self.interval_var.get())
            if interval <= 0:
                interval = 0.6
        except Exception:
            interval = 0.6
        ms = max(10, int(interval * 1000))
        self._board_job = self.after(ms, self._board_tick)

    def start_board(self):
        if self._board_running:
            return
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            if rows <= 0 or cols <= 0:
                raise ValueError("Размеры должны быть положительными числами")
        except Exception as e:
            messagebox.showerror("Ошибка параметров", f"Неверные размеры: {e}")
            return

        try:
            interval = float(self.interval_var.get())
            if interval <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Ошибка параметров", "Интервал должен быть положительным числом (сек).")
            return

        self._update_board_once()
        self._board_running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.randomize_btn.configure(state=tk.DISABLED)
        self._board_tick()

    def stop_board(self):
        if not self._board_running:
            return
        self._board_running = False
        if self._board_job is not None:
            try:
                self.after_cancel(self._board_job)
            except Exception:
                pass
            self._board_job = None
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.randomize_btn.configure(state=tk.NORMAL)

    def save_board(self):
        try:
            content = self.board_text.get("1.0", tk.END)
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.isdir(desktop):
                desktop = os.path.expanduser("~")
            fname = f"numbers_board_{random.randint(0,9999)}.txt"
            path = os.path.join(desktop, fname)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Сохранено", f"Табло сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

    def _on_close(self):
        self.stop_board()
        global random_loop_thread, random_loop_stop_event
        if random_loop_thread and random_loop_thread.is_alive():
            if random_loop_stop_event:
                random_loop_stop_event.set()
            try:
                random_loop_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._egf_thread and self._egf_thread.is_alive():
            if self._egf_stop_event:
                self._egf_stop_event.set()
            try:
                self._egf_thread.join(timeout=1.0)
            except Exception:
                pass
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()
