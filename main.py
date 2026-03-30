from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise ImportError(
        "Не найден пакет Pillow. Установи его командой: pip install pillow"
    ) from exc


# При желании можно вставить координаты, чтобы каждый раз не вводить их вручную.
# Формат: (x, y, w, h)
DEFAULT_TABLE_ROI: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class Detection:
    """Один обнаруженный человек в формате XYXY."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Прототип детекции уборки столика по одному видео."
    )
    parser.add_argument("--video", required=True, help="Путь к входному видео, например video1.mp4")
    parser.add_argument("--output", default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--events_csv", default="events.csv", help="Путь к CSV с событиями")
    parser.add_argument("--report_txt", default="report.txt", help="Путь к краткому текстовому отчету")
    parser.add_argument("--model", default="yolov8n.pt", help="Весы модели Ultralytics")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог confidence для YOLO")
    parser.add_argument("--iou", type=float, default=0.5, help="Порог IoU для YOLO")

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Ручной ROI столика: x y w h",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать окно с обработкой в реальном времени",
    )

    parser.add_argument(
        "--approach_pad_ratio",
        type=float,
        default=0.20,
        help="Насколько расширять зону подхода вокруг столика",
    )
    parser.add_argument(
        "--min_person_area",
        type=int,
        default=6000,
        help="Минимальная площадь bbox человека, чтобы не считать слишком мелкие срабатывания",
    )
    parser.add_argument(
        "--min_person_height",
        type=int,
        default=100,
        help="Минимальная высота bbox человека",
    )
    parser.add_argument(
        "--occupied_overlap",
        type=float,
        default=0.30,
        help="Минимальная доля пересечения bbox человека с ROI столика для состояния 'занят'",
    )
    parser.add_argument(
        "--occupied_roi_overlap",
        type=float,
        default=0.12,
        help="Минимальная доля пересечения ROI столика bbox человека для состояния 'занят'",
    )
    parser.add_argument(
        "--approach_overlap",
        type=float,
        default=0.12,
        help="Минимальная доля пересечения bbox человека с расширенной зоной для события 'подход'",
    )
    parser.add_argument(
        "--approach_roi_overlap",
        type=float,
        default=0.05,
        help="Минимальная доля пересечения расширенной зоны с bbox человека для события 'подход'",
    )
    parser.add_argument(
        "--min_occupied_frames",
        type=int,
        default=4,
        help="Сколько кадров подряд нужно для подтверждения состояния 'занят'",
    )
    parser.add_argument(
        "--min_approach_frames",
        type=int,
        default=4,
        help="Сколько кадров подряд нужно для подтверждения состояния 'подход'",
    )
    parser.add_argument(
        "--min_empty_frames",
        type=int,
        default=4,
        help="Сколько кадров подряд нужно для подтверждения состояния 'пусто'",
    )
    parser.add_argument(
        "--max_missing_frames",
        type=int,
        default=20,
        help="Сколько кадров подряд можно игнорировать пропажу детекции, прежде чем считать стол пустым",
    )
    parser.add_argument(
        "--motion_threshold",
        type=int,
        default=250,
        help="Минимальное число пикселей движения в зоне, чтобы не отпускать occupied при краткой потере детекции",
    )
    parser.add_argument(
        "--motion_diff_value",
        type=int,
        default=18,
        help="Порог для разницы между соседними кадрами при оценке движения",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=24,
        help="Размер шрифта для русского текста на видео",
    )

    return parser.parse_args()


def expand_roi(
    roi: Tuple[int, int, int, int],
    pad_ratio: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """Расширяет ROI на заданный процент от ширины и высоты."""
    x, y, w, h = roi
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width - 1, x + w + pad_x)
    y2 = min(frame_height - 1, y + h + pad_y)

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def rect_to_xyxy(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Переводит прямоугольник из формата x, y, w, h в x1, y1, x2, y2."""
    x, y, w, h = rect
    return x, y, x + w, y + h


def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], state: str) -> None:
    """Рисует ROI столика с цветом, зависящим от текущего состояния."""
    color = {
        "empty": (0, 200, 0),       # зеленый
        "occupied": (0, 0, 255),    # красный
        "approach": (0, 165, 255),  # оранжевый
        "unknown": (0, 255, 255),   # желтый
    }.get(state, (0, 255, 255))

    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    """Вычисляет площадь пересечения двух прямоугольников."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0

    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def bbox_inside_zone(
    person: Detection,
    roi_xyxy: Tuple[int, int, int, int],
    min_person_area: int,
    min_person_height: int,
    min_overlap_person: float,
    min_overlap_roi: float,
) -> bool:
    """
    Проверяет, можно ли считать человека находящимся в зоне.

    Логика строгая:
    - отсекаются слишком маленькие bbox;
    - если центр bbox внутри зоны, считаем попадание уверенным;
    - иначе требуются одновременно достаточные пересечения по bbox человека и по зоне.
    """
    if person.area < min_person_area or person.height < min_person_height:
        return False

    roi_area = (roi_xyxy[2] - roi_xyxy[0]) * (roi_xyxy[3] - roi_xyxy[1])
    if roi_area <= 0:
        return False

    person_xyxy = (person.x1, person.y1, person.x2, person.y2)
    inter_area = intersection_area(person_xyxy, roi_xyxy)
    if inter_area <= 0:
        return False

    overlap_person = inter_area / float(person.area)
    overlap_roi = inter_area / float(roi_area)

    center_x, center_y = person.center
    center_inside = roi_xyxy[0] <= center_x <= roi_xyxy[2] and roi_xyxy[1] <= center_y <= roi_xyxy[3]

    if center_inside:
        return True

    return overlap_person >= min_overlap_person and overlap_roi >= min_overlap_roi


def extract_person_detections(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
) -> List[Detection]:
    """Извлекает детекции людей из кадра через YOLO."""
    results = model.predict(frame, conf=conf, iou=iou, classes=[0], verbose=False)
    detections: List[Detection] = []

    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for box, score in zip(xyxy, confs):
        x1, y1, x2, y2 = map(int, box.tolist())
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(score),
            )
        )

    return detections


def choose_roi(first_frame: np.ndarray, roi_arg: Optional[List[int]]) -> Tuple[int, int, int, int]:
    """
    Выбирает ROI либо из аргумента командной строки, либо из кода, либо мышкой на первом кадре.
    """
    if roi_arg is not None:
        x, y, w, h = roi_arg
        if w <= 0 or h <= 0:
            raise ValueError("Ширина и высота ROI должны быть положительными.")
        return x, y, w, h

    if DEFAULT_TABLE_ROI is not None:
        x, y, w, h = DEFAULT_TABLE_ROI
        if w <= 0 or h <= 0:
            raise ValueError(
                "DEFAULT_TABLE_ROI задан неверно: ширина и высота должны быть положительными."
            )
        print(f"Используется ROI из кода: {DEFAULT_TABLE_ROI}")
        return DEFAULT_TABLE_ROI

    print("Выдели мышкой один столик на первом кадре и нажми ENTER или SPACE.")
    selected = cv2.selectROI("Выбор ROI столика", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Выбор ROI столика")

    x, y, w, h = map(int, selected)
    if w <= 0 or h <= 0:
        raise ValueError("ROI не выбран. Запусти скрипт заново и выдели область столика.")
    return x, y, w, h


def build_event_row(
    timestamp_sec: float,
    frame_idx: int,
    event_type: str,
    state: str,
    detected_people: int,
    occupied_people: int,
    approach_people: int,
    motion_pixels: int,
    note: str = "",
) -> dict:
    """Формирует строку для таблицы событий."""
    return {
        "timestamp_sec": round(timestamp_sec, 3),
        "frame_idx": frame_idx,
        "event_type": event_type,
        "state": state,
        "detected_people": detected_people,
        "occupied_people": occupied_people,
        "approach_people": approach_people,
        "motion_pixels": motion_pixels,
        "note": note,
    }


def calculate_average_delay(events_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Считает среднюю задержку между событием 'empty' и следующим событием 'approach'.

    Используются только подтвержденные события переходов из events_df.
    """
    if events_df.empty:
        return float("nan"), pd.DataFrame()

    approach_df = events_df[events_df["event_type"] == "approach"].copy()
    empty_df = events_df[events_df["event_type"] == "empty"].copy()

    if empty_df.empty or approach_df.empty:
        return float("nan"), pd.DataFrame()

    pair_rows = []
    empty_times = empty_df["timestamp_sec"].tolist()
    approach_times = approach_df["timestamp_sec"].tolist()

    approach_index = 0
    for empty_time in empty_times:
        while approach_index < len(approach_times) and approach_times[approach_index] <= empty_time:
            approach_index += 1

        if approach_index >= len(approach_times):
            break

        delay = approach_times[approach_index] - empty_time
        pair_rows.append(
            {
                "table_empty_at_sec": round(empty_time, 3),
                "next_approach_at_sec": round(approach_times[approach_index], 3),
                "delay_sec": round(delay, 3),
            }
        )
        approach_index += 1

    if not pair_rows:
        return float("nan"), pd.DataFrame()

    pair_df = pd.DataFrame(pair_rows)
    return float(pair_df["delay_sec"].mean()), pair_df


def find_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Ищет шрифт с поддержкой кириллицы.
    Сначала пробует Windows, затем типовые Linux-пути.
    """
    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in possible_fonts:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)

    raise FileNotFoundError(
        "Не найден шрифт с поддержкой кириллицы. "
        "Поставь Arial или DejaVuSans либо укажи путь к шрифту вручную."
    )


def draw_overlay_text(
    frame: np.ndarray,
    timestamp_sec: float,
    stable_state: str,
    observed_state: str,
    detected_people: int,
    transition_counter: int,
    font_size: int,
) -> None:
    """
    Рисует служебную текстовую информацию поверх кадра.
    """
    state_map = {
        "empty": "пусто",
        "occupied": "занято",
        "approach": "подход",
        "unknown": "неизвестно",
    }

    stable_state_ru = state_map.get(stable_state, stable_state)
    observed_state_ru = state_map.get(observed_state, observed_state)

    lines = [
        f"Время: {timestamp_sec:.2f} c",
        f"Стабильное состояние: {stable_state_ru}",
        f"Наблюдаемое состояние: {observed_state_ru}",
        f"Людей в зоне: {detected_people}",
        f"Счетчик подтверждения: {transition_counter}",
    ]

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = find_font(font_size)

    x, y = 20, 20
    line_step = int(font_size * 1.35)

    for idx, line in enumerate(lines):
        draw.text((x, y + idx * line_step), line, font=font, fill=(255, 255, 255))

    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def confirm_frames_required(state: str, args: argparse.Namespace) -> int:
    """Возвращает число кадров, нужное для подтверждения конкретного состояния."""
    mapping = {
        "empty": args.min_empty_frames,
        "approach": args.min_approach_frames,
        "occupied": args.min_occupied_frames,
    }
    return mapping.get(state, args.min_empty_frames)


def log_transition(
    event_rows: List[dict],
    timestamp_sec: float,
    frame_idx: int,
    event_type: str,
    state: str,
    detected_people: int,
    occupied_people: int,
    approach_people: int,
    motion_pixels: int,
    note: str,
) -> None:
    """Добавляет событие в журнал."""
    event_rows.append(
        build_event_row(
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
            event_type=event_type,
            state=state,
            detected_people=detected_people,
            occupied_people=occupied_people,
            approach_people=approach_people,
            motion_pixels=motion_pixels,
            note=note,
        )
    )


def compute_motion_pixels(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    roi_xyxy: Tuple[int, int, int, int],
    diff_value: int,
) -> int:
    """
    Оценивает количество пикселей движения внутри ROI по разнице соседних кадров.
    Это помогает не отпускать occupied при краткой потере детекции.
    """
    if prev_gray is None:
        return 0

    diff = cv2.absdiff(prev_gray, gray)
    x1, y1, x2, y2 = roi_xyxy
    roi_diff = diff[y1:y2, x1:x2]

    if roi_diff.size == 0:
        return 0

    _, binary = cv2.threshold(roi_diff, diff_value, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 5)
    binary = cv2.dilate(binary, None, iterations=1)

    return int(cv2.countNonZero(binary))


def main() -> None:
    """Основная точка входа: чтение видео, детекция, логирование событий и сохранение результата."""
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    output_path = Path(args.output)
    events_csv_path = Path(args.events_csv)
    report_txt_path = Path(args.report_txt)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр из видео.")

    table_roi = choose_roi(first_frame, args.roi)
    frame_h, frame_w = first_frame.shape[:2]
    approach_roi = expand_roi(
        table_roi,
        pad_ratio=args.approach_pad_ratio,
        frame_width=frame_w,
        frame_height=frame_h,
    )

    table_roi_xyxy = rect_to_xyxy(table_roi)
    approach_roi_xyxy = rect_to_xyxy(approach_roi)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or first_frame.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {output_path}")

    try:
        model = YOLO(args.model)
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить модель '{args.model}'. "
            f"Проверь наличие файла весов и установку ultralytics."
        ) from exc

    event_rows: List[dict] = []

    # Машина состояний.
    stable_state = "unknown"
    candidate_raw_state: Optional[str] = None
    candidate_count = 0

    # Сколько кадров подряд человек не был виден или не было движения в зоне.
    absence_counter = 0

    # Предыдущий кадр для оценки движения.
    prev_gray: Optional[np.ndarray] = None

    frame_idx = 0

    def process_frame(frame: np.ndarray, current_frame_idx: int) -> None:
        """
        Обрабатывает один кадр:
        - извлекает детекции людей;
        - определяет наблюдаемое состояние столика;
        - не позволяет occupied перейти напрямую в approach;
        - добавляет инерцию против краткой потери детекции;
        - использует движение как дополнительный сигнал;
        - подтверждает состояние по нескольким кадрам;
        - пишет события в список;
        - сохраняет кадр в выходное видео.
        """
        nonlocal stable_state, candidate_raw_state, candidate_count, absence_counter, prev_gray

        timestamp_sec = current_frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_pixels = compute_motion_pixels(
            prev_gray=prev_gray,
            gray=gray,
            roi_xyxy=approach_roi_xyxy,
            diff_value=args.motion_diff_value,
        )
        prev_gray = gray

        detections = extract_person_detections(model, frame, conf=args.conf, iou=args.iou)

        people_in_occupied_zone = [
            det for det in detections
            if bbox_inside_zone(
                person=det,
                roi_xyxy=table_roi_xyxy,
                min_person_area=args.min_person_area,
                min_person_height=args.min_person_height,
                min_overlap_person=args.occupied_overlap,
                min_overlap_roi=args.occupied_roi_overlap,
            )
        ]

        people_in_approach_zone = [
            det for det in detections
            if bbox_inside_zone(
                person=det,
                roi_xyxy=approach_roi_xyxy,
                min_person_area=args.min_person_area,
                min_person_height=args.min_person_height,
                min_overlap_person=args.approach_overlap,
                min_overlap_roi=args.approach_roi_overlap,
            )
        ]

        occupied_observed = len(people_in_occupied_zone) > 0
        approach_observed = len(people_in_approach_zone) > 0 and not occupied_observed
        motion_observed = motion_pixels >= args.motion_threshold

        # Если человек видим или есть движение, сбрасываем счетчик пропажи.
        if occupied_observed or motion_observed:
            absence_counter = 0
        else:
            absence_counter += 1

        # Наблюдаемое состояние.
        # Главное правило: если стол уже occupied, мы НЕ разрешаем ему
        # напрямую уходить в approach. Сначала удерживаем occupied,
        # а только после подтвержденного освобождения переходим в empty.
        if occupied_observed:
            observed_state = "occupied"

        elif stable_state == "occupied":
            # Пока стол был занят, краткую потерю человека не считаем уходом.
            if motion_observed or absence_counter <= args.max_missing_frames:
                observed_state = "occupied"
            else:
                observed_state = "empty"

        elif approach_observed:
            observed_state = "approach"

        elif stable_state == "approach":
            if motion_observed or absence_counter <= args.max_missing_frames:
                observed_state = "approach"
            else:
                observed_state = "empty"

        else:
            observed_state = "empty"

        if candidate_raw_state != observed_state:
            candidate_raw_state = observed_state
            candidate_count = 1
        else:
            candidate_count += 1

        required_frames = confirm_frames_required(candidate_raw_state, args)

        # Первое устойчивое состояние.
        if stable_state == "unknown":
            if candidate_count >= required_frames:
                stable_state = candidate_raw_state or "empty"
                log_transition(
                    event_rows=event_rows,
                    timestamp_sec=timestamp_sec,
                    frame_idx=current_frame_idx,
                    event_type=f"initial_{stable_state}",
                    state=stable_state,
                    detected_people=len(detections),
                    occupied_people=len(people_in_occupied_zone),
                    approach_people=len(people_in_approach_zone),
                    motion_pixels=motion_pixels,
                    note="Первое подтвержденное состояние",
                )
                candidate_count = 0

        # Дальнейшие переходы состояний.
        else:
            if candidate_raw_state == stable_state:
                candidate_count = 0
            elif candidate_count >= required_frames:
                prev_state = stable_state
                new_state = candidate_raw_state or "empty"

                # empty -> approach
                if prev_state == "empty" and new_state == "approach":
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="approach",
                        state="approach",
                        detected_people=len(detections),
                        occupied_people=len(people_in_occupied_zone),
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Человек появился рядом со столиком после периода пустоты",
                    )

                # empty -> occupied
                elif prev_state == "empty" and new_state == "occupied":
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="approach",
                        state="approach",
                        detected_people=len(detections),
                        occupied_people=len(people_in_occupied_zone),
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Человек подошел к столику и сразу попал в зону занятия",
                    )
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="occupied",
                        state="occupied",
                        detected_people=len(detections),
                        occupied_people=len(people_in_occupied_zone),
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Столик стал занят",
                    )

                # approach -> occupied
                elif prev_state == "approach" and new_state == "occupied":
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="occupied",
                        state="occupied",
                        detected_people=len(detections),
                        occupied_people=len(people_in_occupied_zone),
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Человек дошел до столика и остается в зоне",
                    )

                # occupied -> empty
                elif prev_state == "occupied" and new_state == "empty":
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="empty",
                        state="empty",
                        detected_people=len(detections),
                        occupied_people=0,
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Столик стал пустым",
                    )

                # approach -> empty
                elif prev_state == "approach" and new_state == "empty":
                    log_transition(
                        event_rows=event_rows,
                        timestamp_sec=timestamp_sec,
                        frame_idx=current_frame_idx,
                        event_type="empty",
                        state="empty",
                        detected_people=len(detections),
                        occupied_people=0,
                        approach_people=len(people_in_approach_zone),
                        motion_pixels=motion_pixels,
                        note="Подход был кратким, затем зона снова опустела",
                    )

                stable_state = new_state
                candidate_count = 0


        draw_roi(frame, table_roi, stable_state)
        draw_overlay_text(
            frame=frame,
            timestamp_sec=timestamp_sec,
            stable_state=stable_state,
            observed_state=observed_state,
            detected_people=len(detections),
            transition_counter=candidate_count,
            font_size=args.font_size,
        )

        writer.write(frame)

        if args.show:
            cv2.imshow("Прототип детекции уборки столика", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("Остановка по нажатию клавиши q.")

    try:
        process_frame(first_frame, frame_idx)
        frame_idx += 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            process_frame(frame, frame_idx)
            frame_idx += 1

    except KeyboardInterrupt:
        print("Обработка остановлена пользователем.")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    events_df = pd.DataFrame(event_rows)
    events_df.to_csv(events_csv_path, index=False)

    average_delay_sec, pair_df = calculate_average_delay(events_df)

    report_lines = [
        f"Видео: {video_path}",
        f"Выходное видео: {output_path}",
        f"CSV с событиями: {events_csv_path}",
        f"ROI столика (x, y, w, h): {table_roi}",
        f"Расширенная зона подхода (x, y, w, h): {approach_roi}",
        f"Количество обработанных кадров: {frame_idx}",
    ]

    if math.isnan(average_delay_sec):
        report_lines.append("Средняя задержка: недостаточно пар empty -> approach для расчета.")
    else:
        report_lines.append(f"Средняя задержка между уходом и следующим подходом: {average_delay_sec:.3f} c")

    report_txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== Обработка завершена ===")
    for line in report_lines:
        print(line)

    if not events_df.empty:
        print("\nЖурнал событий:")
        print(events_df.to_string(index=False))
    else:
        print("\nСобытия не были зафиксированы.")

    if not pair_df.empty:
        print("\nПары для расчета задержки:")
        print(pair_df.to_string(index=False))



if __name__ == "__main__":
    main()