import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import random
import time
from pydub import AudioSegment
import simpleaudio as sa

# Параметры (подберите под себя)
CHUNK_DURATION_MS = 3000      # длина одного воспроизводимого чанка (мс)
CHUNK_OVERLAP_MS = 200        # на сколько миллисекунд стартуем следующий чанк раньше (оверлап)
PIECE_MIN_MS = 150            # минимальная длина нарезки в чанке
PIECE_MAX_MS = 700            # максимальная длина нарезки в чанке
INNER_CROSSFADE_MS = 60      # кроссфейд между кусочками внутри чанка

# Глобальные переменные
selected_segments = []   # список загруженных AudioSegment
selected_paths = []      # список путей (для отображения)
playing = False
play_thread = None

# GUI
root = tk.Tk()
root.title("Бесконечный ЭГФ (без пауз)")
root.geometry("600x420")

audio_listbox = tk.Listbox(root, width=80, height=12)
audio_listbox.pack(pady=10)

def select_audio():
    file_path = filedialog.askopenfilename(
        title="Выберите аудио файл",
        filetypes=[("Audio files", "*.mp3;*.wav;*.ogg;*.flac;*.m4a")]
    )
    if not file_path:
        return
    try:
        seg = AudioSegment.from_file(file_path)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")
        return
    selected_segments.append(seg)
    selected_paths.append(file_path)
    audio_listbox.insert(tk.END, file_path)

def build_chunk():
    """
    Создает один AudioSegment-чанк, состоящий из случайных нарезок с кроссфейдом.
    """
    if not selected_segments:
        return None
    combined = AudioSegment.silent(duration=0)
    remaining = CHUNK_DURATION_MS
    # Собираем кусочки, пока не наберём длительность чанка
    while remaining > 0:
        piece_len = random.randint(PIECE_MIN_MS, min(PIECE_MAX_MS, remaining))
        src = random.choice(selected_segments)
        if len(src) <= piece_len:
            piece = src[:]  # весь файл, если короче
        else:
            start = random.randint(0, len(src) - piece_len)
            piece = src[start:start + piece_len]
        # первый кусок — просто добавляем, следующие — с кроссфейдом
        if len(combined) == 0:
            combined = piece
        else:
            # append с кроссфейдом — сглаживает стыки
            try:
                combined = combined.append(piece, crossfade=INNER_CROSSFADE_MS)
            except TypeError:
                # старые версии pydub могут не поддерживать append(..., crossfade=)
                combined = combined + piece
        remaining = CHUNK_DURATION_MS - len(combined)
        # защитный break на случай, если комбинированный сегмент вырос больше
        if len(combined) >= CHUNK_DURATION_MS:
            break
    # Обрежем лишнее, если получилось длиннее
    combined = combined[:CHUNK_DURATION_MS]
    return combined

def play_loop():
    """
    Генерирует чанки и запускает их воспроизведение с перекрытием, чтобы не было пауз.
    """
    global playing
    # Подготовка: если нет сегментов — выходим
    if not selected_segments:
        playing = False
        btn_start.config(text="Генерировать ЭГФ")
        messagebox.showerror("Ошибка", "Выберите хотя бы один аудио файл.")
        return

    # Немного предгенерации первого чанка
    next_chunk = build_chunk()
    if next_chunk is None:
        playing = False
        btn_start.config(text="Генерировать ЭГФ")
        return

    # Стартуем цикл: каждую итерацию стартуем воспроизведение next_chunk
    while playing:
        # Запускаем воспроизведение текущего чанка (не блокируя поток)
        try:
            raw = next_chunk.raw_data
            channels = next_chunk.channels
            sample_width = next_chunk.sample_width
            frame_rate = next_chunk.frame_rate
            play_obj = sa.play_buffer(raw, channels, sample_width, frame_rate)
        except Exception as e:
            print("Ошибка воспроизведения:", e)
            # если что-то сломалось — попробуем дождаться небольшую паузу и продолжить
            time.sleep(0.1)
            if not playing:
                break
            else:
                continue

        # Параллельно генерируем следующий чанк (чтобы успеть до старта)
        # Здесь мы генерируем заранее, чтобы минимизировать задержки
        following_chunk = build_chunk()

        # Ждём до момента запуска следующего чанка:
        # время ожидания = длительность чанка - overlap
        sleep_time = max(0, (CHUNK_DURATION_MS - CHUNK_OVERLAP_MS) / 1000.0)
        # Но если генерация following_chunk заняла долго — не блокируем слишком долго
        t0 = time.time()
        time.sleep(sleep_time)
        # к моменту следующей итерации next_chunk будет following_chunk
        next_chunk = following_chunk if following_chunk is not None else build_chunk()

        # Если play_obj ещё играет и мы собираемся оверлэпать — не ждём .wait_done()
        # Это обеспечивает наложение и отсутствие тишины между чанками.

    # при выходе — ничего дополнительно не требуется (воспроизведения закончатся сами)
    playing = False
    btn_start.config(text="Генерировать ЭГФ")

def start_stop():
    global playing, play_thread
    if not playing:
        if not selected_segments:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите хотя бы один аудио файл.")
            return
        playing = True
        btn_start.config(text="Остановить")
        # Запускаем поток воспроизведения
        play_thread = threading.Thread(target=play_loop, daemon=True)
        play_thread.start()
    else:
        # Останавливаем поток — флаг остановки. Текущие play_buffer-объекты допроиграют свои буферы, но новых не будет.
        playing = False
        btn_start.config(text="Остановить...")  # индикация процесса остановки

# GUI-кнопки
btn_select = tk.Button(root, text="Выбрать аудио", command=select_audio)
btn_select.pack(pady=6)

btn_start = tk.Button(root, text="Генерировать ЭГФ", command=start_stop)
btn_start.pack(pady=6)

root.mainloop()
