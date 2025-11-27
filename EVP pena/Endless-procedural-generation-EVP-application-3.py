import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import struct
import threading
import random
import time
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
            arr = arr.reshape(-1, 1)
            if arr.shape[1] != channels:
                arr = np.tile(arr, (1, channels))
    return arr.astype(np.float32)


def generate_egf_tone(number, sample_rate=DEFAULT_SAMPLE_RATE, duration=0.18, channels=1):
    """
    Генерирует короткий тон с экспоненциальной огибающей ("ЭГФ" — эксп. затухание) для заданного числа 0..255.
    По умолчанию: линейное отображение числа в диапазон частот 200..5000 Hz.
    Возвращает numpy array dtype float32 готовый для sounddevice.
    """
    # Ограничиваем вход
    n = max(0, min(255, int(number)))
    # Частота: линейно 200..5000 Гц
    f_min = 200.0
    f_max = 5000.0
    freq = f_min + (f_max - f_min) * (n / 255.0)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # экспоненциальная огибающая: exp(-t/tau)
    tau = duration / 4.0
    env = np.exp(-t / tau)
    wave = np.sin(2.0 * np.pi * freq * t) * env

    # моно -> нужное число каналов
    if channels == 1:
        out = wave.astype(np.float32)
    else:
        out = np.repeat(wave[:, None], channels, axis=1).astype(np.float32)
    # нормализация (уже в -1..1 диапазоне)
    return out


def generate_egf_bytes(number, sample_rate=DEFAULT_SAMPLE_RATE, duration=0.18, channels=1, sample_width=1):
    """
    Генерирует звук в режиме #META mode=bytes: создаём буфер байтов, заполненный значением number (0..255),
    конвертируем в float32 через _bytes_to_float32 (имитируя 8-bit PCM), затем применяем экспоненциальную огибающую.
    Это позволяет воспроизводить "EGF" на аудиопотоке, совместимом с mode=bytes.
    Возвращает numpy float32 готовый для sounddevice.
    """
    n = max(0, min(255, int(number)))
    total_samples = max(1, int(sample_rate * duration))
    # создаём байтовый буфер: каждый сэмпл = значение n
    raw = bytes([n]) * total_samples
    audio = _bytes_to_float32(raw, sample_width=sample_width, channels=channels)

    # применим экспоненциальную огибающую к float-представлению
    t = np.linspace(0, duration, total_samples, endpoint=False)
    tau = duration / 4.0
    env = np.exp(-t / tau).astype(np.float32)

    if channels == 1:
        audio = audio * env
    else:
        # если мультиканал, масштабируем каждый канал одинаково
        try:
            audio = (audio.T * env).T
        except Exception:
            audio = audio * env[:, None]

    return audio.astype(np.float32)


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


# --- GUI: комбинированное приложение (звуки + табло чисел 0..255) ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Конвертер чисел <-> MP3 + Табло 0..255 + EGF")
        self.geometry("1280x760")

        info = ("В этом окне можно: загрузить файл с числами (0..255) и запустить бесконечный\n"
                "циклический поток случайных фрагментов из файла (gapless через sounddevice),\n"
                "запустить бесконечную трансляцию ЭГФ (генерируемая огибающей экспоненциально в режиме bytes),\n"
                "а также открыть табло с числами 0..255 в виде сетки (настраиваемые строки/столбцы,\n"
                "последовательный или перемешанный порядок, автогенерация).")

        tk.Label(self, text=info, justify=tk.LEFT).pack(pady=8)

        # Верхняя панель: поле ввода (для парсинга), кнопки управления звуком
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10)

        self.text_input = tk.Text(top_frame, height=6, width=120)
        self.text_input.pack(side=tk.LEFT, padx=(0,8), pady=6)
        self.text_input.insert("1.0", "Пример содержимого файла:\n#META mode=bytes sample_rate=44100 channels=1 sample_width=1\n0 127 255 34 200 ...")

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

        # Новая кнопка: бесконечная трансляция ЭГФ
        self.egf_btn = tk.Button(btns_frame, text="EVP трансляция (Старт)", command=self.on_egf_toggle, width=30)
        self.egf_btn.pack(pady=8)

        # Разделитель
        sep = tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN)
        sep.pack(fill=tk.X, padx=5, pady=8)

        # Табло чисел 0..255
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

        # Текстовое поле для табло
        self.board_text = tk.Text(board_frame, wrap=tk.NONE, font=("Consolas", 12))
        self.board_text.pack(fill=tk.BOTH, expand=True)
        self.board_text.configure(state=tk.DISABLED)

        # Внутренние
        self._board_running = False
        self._board_job = None
        self._board_numbers = list(range(256))

        # EGF stream internals
        self._egf_thread = None
        self._egf_stop_event = None
        self._egf_running = False

        # Init board
        self._init_board()
        self._render_board()

        # Обработчик закрытия
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

    # --- EGF stream: бесконечная трансляция с подсветкой чисел ---
    def on_egf_toggle(self):
        if self._egf_running:
            # остановка
            self._egf_stop_event.set()
            try:
                sd.stop()
            except Exception:
                pass
            if self._egf_thread:
                self._egf_thread.join(timeout=1.0)
            self._egf_thread = None
            self._egf_stop_event = None
            self._egf_running = False
            self.egf_btn.config(text="EVP трансляция (Старт)")
            # убрать подсветку
            try:
                self.board_text.tag_remove("egf_highlight", "1.0", tk.END)
            except Exception:
                pass
            return

        # старт
        ev = threading.Event()
        self._egf_stop_event = ev
        t = threading.Thread(target=self._egf_loop_bytes_mode, args=(ev,), daemon=True)
        self._egf_thread = t
        self._egf_running = True
        t.start()
        self.egf_btn.config(text="EGF трансляция (Стоп)")

    def _egf_loop_bytes_mode(self, stop_event):
        """
        Теперь EGF генерируется в режиме bytes: для каждого случайного числа создаём буфер байтов
        (значение повторяется), конвертируем его в float32 через _bytes_to_float32 и применяем экспоненциальную огибающую.
        Это даёт более "bytes-совместимый" звук, как вы просили (#META mode=bytes ...).
        """
        sr = DEFAULT_SAMPLE_RATE
        channels = DEFAULT_CHANNELS
        sample_width = DEFAULT_SAMPLE_WIDTH
        duration = 0.16
        while not stop_event.is_set():
            n = random.randint(0, 255)
            # подсветка в главном потоке
            try:
                self.after(0, lambda n=n: self._highlight_number(n))
            except Exception:
                pass

            try:
                audio = generate_egf_bytes(n, sample_rate=sr, duration=duration, channels=channels, sample_width=sample_width)
                sd.play(audio, samplerate=sr)
                sd.wait()
            except Exception:
                if stop_event.wait(0.05):
                    break
                continue

        # при выходе убираем подсветку
        try:
            self.after(0, lambda: self.board_text.tag_remove("egf_highlight", "1.0", tk.END))
        except Exception:
            pass

    def _highlight_number(self, n):
        """
        Подсвечивает число n на текущем табло (если оно отображается).
        Работает для формата, где каждое поле занимает 3 символа и разделитель — один пробел.
        """
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
            # число не отображается в текущей сетке
            return
        r = idx // cols
        c = idx % cols
        start_char = c * 4  # 3 символа + 1 пробел
        line = r + 1
        start = f"{line}.{start_char}"
        end = f"{line}.{start_char + 3}"
        try:
            # удаляем предыдущую подсветку и ставим новую
            self.board_text.tag_remove("egf_highlight", "1.0", tk.END)
            self.board_text.tag_add("egf_highlight", start, end)
            self.board_text.tag_config("egf_highlight", background="yellow")
            # обеспечить видимость
            self.board_text.see(start)
        except Exception:
            pass

    # --- board (табло 0..255) ---
    def _init_board(self):
        # по умолчанию последовательные 0..255
        self._board_numbers = list(range(256))

    def _render_board(self):
        rows = max(1, int(self.rows_var.get()))
        cols = max(1, int(self.cols_var.get()))
        nums = self._board_numbers

        # если размера сетки меньше 256 — показываем первые rows*cols чисел; если больше — дополняем пустыми
        total = rows * cols
        if total <= 0:
            lines = [""]
        else:
            if len(nums) < total:
                # дополним пустыми строками
                nums2 = nums + [''] * (total - len(nums))
            else:
                nums2 = nums[:total]

            lines = []
            for r in range(rows):
                row_slice = nums2[r * cols:(r + 1) * cols]
                # форматируем: числа выравниваем по правому краю, ширина 3
                formatted = [f"{n:3}" if isinstance(n, int) else '   ' for n in row_slice]
                lines.append(" ".join(formatted))

        text_out = "\n".join(lines)
        self.board_text.configure(state=tk.NORMAL)
        self.board_text.delete("1.0", tk.END)
        self.board_text.insert(tk.END, text_out)
        self.board_text.configure(state=tk.DISABLED)

    def _update_board_once(self):
        # обновление — для режима shuffle мы просто перемешиваем список
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
        # останавливаем табло
        self.stop_board()
        # останавливаем аудиопотоки если запущены
        global random_loop_thread, random_loop_stop_event
        if random_loop_thread and random_loop_thread.is_alive():
            if random_loop_stop_event:
                random_loop_stop_event.set()
            try:
                random_loop_thread.join(timeout=1.0)
            except Exception:
                pass
        # останавливаем EGF
        if self._egf_thread and self._egf_thread.is_alive():
            if self._egf_stop_event:
                self._egf_stop_event.set()
            try:
                sd.stop()
            except Exception:
                pass
            try:
                self._egf_thread.join(timeout=1.0)
            except Exception:
                pass
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()
