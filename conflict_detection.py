patients = [
    ("Patient 1", "COVID-19", "COVID-19"),
    ("Patient 2", "Pneumonia", "Normal"),
    ("Patient 3", "Lung Opacity", "Lung Opacity"),
    ("Patient 4", "Normal", "COVID-19")
]

print("=== Clinical VLM Conflict Detection ===\n")

for patient, image_pred, multimodal_pred in patients:

    print(f"Patient: {patient}")
    print(f"Image Only Prediction: {image_pred}")
    print(f"Image + Clinical Data Prediction: {multimodal_pred}")

    if image_pred != multimodal_pred:
        print("Result: CONFLICT DETECTED")
    else:
        print("Result: NO CONFLICT")

    print("-" * 40)