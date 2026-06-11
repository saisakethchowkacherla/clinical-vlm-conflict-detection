import random

samples = 1000

conflicts = 0
correct_raw = 0
correct_defer = 0

for _ in range(samples):

    true_label = random.choice(
        ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]
    )

    image_only = random.choice(
        ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]
    )

    image_lab = random.choice(
        ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]
    )

    if image_only != image_lab:
        conflicts += 1

    if image_only == true_label:
        correct_raw += 1

    if image_lab == true_label:
        correct_defer += 1

raw_accuracy = correct_raw / samples
defer_accuracy = correct_defer / samples
conflict_rate = conflicts / samples

print("Raw Accuracy:", round(raw_accuracy, 3))
print("Defer Accuracy:", round(defer_accuracy, 3))
print("Conflict Rate:", round(conflict_rate, 3))