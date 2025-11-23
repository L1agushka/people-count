import os
import cv2
import csv
import time
from ultralytics import YOLO
from lwcc import LWCC

TEST_DIR = "test_images"
OUTPUT_CSV = "submission.csv"
yolo_model = YOLO("yolov8l.pt")
dm_model = LWCC.load_model(model_name="DM-Count", model_weights="SHB")

def is_teacher_box(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect_ratio = h / w  # вертикальность
    if aspect_ratio > 2.0:
        return True
    return False

def count_yolo_and_teacher(img, conf_threshold=0.60): # подсчет всех через yolo и отсечение лишних по геометрическому условию
    results = yolo_model(img)
    total_people = 0
    teacher_count = 0
    for r in results:
        boxes = r.boxes
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        for cls, conf, box in zip(classes, confs, xyxy):
            if cls != 0:
                continue
            if conf < conf_threshold:
                continue
            total_people += 1
            if is_teacher_box(box):
                teacher_count += 1
    return total_people, teacher_count

results = []
processing_times = []

print("Начало обработки")
start_time_total = time.time()

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    img_path = os.path.join(TEST_DIR, fname)
    start_time = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print("Не удалось загрузить:", img_path)
        continue
    yolo_total, teachers = count_yolo_and_teacher(img)
    dm_count = LWCC.get_count(img_path, model_name="DM-Count", model_weights="SHA")
    ensemble_raw = 0.0 * yolo_total + 1 * dm_count # увеличивать вес yolo при разреженных сценах
    final_count = ensemble_raw - teachers
    final_count = round(final_count)
    img_id = os.path.splitext(fname)[0]
    results.append({
        "IMG_ID": img_id,
        "label": final_count
    })
    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)

end_time_total = time.time()
total_time = end_time_total - start_time_total
avg_time_per_image = total_time / len(results) if results else 0
min_time = min(processing_times) if processing_times else 0
max_time = max(processing_times) if processing_times else 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["IMG_ID", "label"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("результаты сохранены в", OUTPUT_CSV)

print(f"   • Общее время обработки: {total_time:.2f} секунд")
print(f"   • Количество обработанных изображений: {len(results)}")
print(f"   • Среднее время на одно изображение: {avg_time_per_image:.3f} сек")
print(f"   • Минимальное время обработки: {min_time:.3f} сек")
print(f"   • Максимальное время обработки: {max_time:.3f} сек")
print(f"   • Скорость обработки: {1 / avg_time_per_image:.2f} изображений/сек")
