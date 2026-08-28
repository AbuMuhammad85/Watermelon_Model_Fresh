# Noma AI - Flutter Integration Contract (Watermelon Leaf Diagnostics)

This package contains the final offline AI assets and specifications required to implement the watermelon disease diagnostic pipeline inside the Flutter mobile application.

---

## 1. Package File Contents

* **`model_float16.tflite`**: The final Float16 TFLite classifier model.
* **`labels.txt`**: The target class labels in alphabetical order (matching output indexes 0 to 3).
* **`disease_knowledge_base.json`**: Complete offline symptoms, causes, prevention, treatments, translations (English and Hausa), and recommendation engine routing tables.

---

## 2. Classification Classes & Output Mapping

The model outputs a 1D tensor of 4 probabilities representing:

| Index | Class Name | English Display Name | Hausa Display Name |
| :--- | :--- | :--- | :--- |
| **0** | `Anthracnose` | Anthracnose | Ciwon Digo-digo (Anthracnose) |
| **1** | `Downy_Mildew` | Downy Mildew | Ciwon Downy Mildew (Farin Kura) |
| **2** | `Healthy` | Healthy | Lafiyayyen Ganye |
| **3** | `Mosaic_Virus` | Mosaic Virus | Ciwon Mosaic Virus |

---

## 3. TFLite Tensor Specifications

* **Input Tensor Shape:** `[1, 224, 224, 3]` (Batch size 1, Width 224, Height 224, Channels RGB).
* **Input Data Type:** `Float32` (4 bytes per element).
* **Output Tensor Shape:** `[1, 4]` (Batch size 1, 4 class probabilities).
* **Output Data Type:** `Float32`.

> [!IMPORTANT]
> **Input Pixel Value Range:**
> Pixel values must be in the range **`[0.0, 255.0]`**, NOT normalised to `[0, 1]` or `[-1, 1]`. 
> MobileNetV3-Small handles input scaling internally inside its model architecture graph. Scaling inputs in Dart will cause wrong diagnoses.

---

## 4. Image Pre-Inference Pipeline (Dart Implementation)

To replicate the offline Python pipeline, Flutter must execute the following operations in order on the raw camera/gallery image before running inference:

### Step A: Offline Image Quality-Gate
Before running inference, verify the image quality. If a check fails, warn the user and recommend a retake instead of showing a diagnostic result.

1. **Blur Detection (Laplacian Variance):**
   * Convert the input image to grayscale.
   * Convolve it with a standard 3x3 Laplacian filter kernel:
     ```
     [ 0,  1,  0]
     [ 1, -4,  1]
     [ 0,  1,  0]
     ```
   * Compute the variance of the convolved image.
   * **Rule:** If variance is **`< 60.0`**, fail the gate with reason `blurry`.
2. **Exposure Check (Grayscale Mean):**
   * Compute the average intensity of the grayscale pixels.
   * **Rule:** If mean is **`< 40.0`**, fail with reason `too_dark`. If mean is **`> 235.0`**, fail with reason `too_bright`.
3. **Framing & Size Check (Excess Green ratio):**
   * For each pixel, compute the Excess Green Index: `ExG = 2 * Green - Red - Blue`.
   * Count pixels where `ExG > 15.0` and `Green > 35.0` (indicates leaf pixels).
   * **Rule:** If the ratio of leaf pixels to total pixels is **`< 0.20 (20%)`**, fail with reason `leaf_too_small` (ask the user to move closer / zoom in).

### Step B: Bounding-Box Leaf Crop
To strip out background soil, weeds, and resolve far-away leaf issues:
1. Identify all pixels matching `ExG > 12.0` and `Green > 30.0`.
2. To remove single high-frequency noise pixels (e.g. soil texture), apply a simple noise filter (e.g. discard green pixels that have no green neighbors in a 3x3 grid).
3. Find the minimum and maximum X and Y coordinates of the remaining leaf pixels to define the leaf bounding box: `[xmin, ymin, xmax, ymax]`.
4. Add a padding margin of 12 pixels around the box (clamping to the image boundaries).
5. Crop the original image to this bounding box.
6. Resize the cropped crop to `224x224` pixels using bilinear interpolation.
7. Convert the pixels to float32 values in `[0.0, 255.0]` to feed the input tensor.

---

## 5. Post-Inference Uncertainty Handling

When predictions are returned from the TFLite interpreter, evaluate confidence before showing a diagnosis:
* **Confidence Gate Threshold:** `min_confidence = 0.65 (65%)`.
* **Margin Gate Threshold:** `min_margin = 0.15 (15%)` (difference between the top 1 and top 2 highest probabilities).

**Inference Routing Rules:**
1. Sort probabilities in descending order.
2. If `top_probability < 0.65` OR `(top_probability - second_probability) < 0.15`:
   * Set status to `LOW_CONFIDENCE`.
   * Do not display a disease diagnosis to the farmer.
   * Instead, display the warning: *"Low diagnostic confidence. Please retake the photo. Ensure the leaf is centered, flat, and has good indirect light."* (Or Hausa equivalent).
3. Otherwise, set status to `HIGH_CONFIDENCE` and display the diagnosed condition.

---

## 6. Severity Concept Mapping

The app must include a severity selector (Low, Moderate, High) in the UI. 
* To assist the farmer, the app can estimate severity visually using the color-based ratio of necrotic spots to total leaf area:
  * Count green pixels: `green_mask = (ExG > 15.0) & (Green > 35.0)`
  * Count brown/yellow spots: `necrotic_mask = (Red > Blue) & (Green > Blue) & (ExG <= 15.0) & (Red > 35.0) & (Green > 35.0)`
  * Compute `ratio = necrotic_pixels / (green_pixels + necrotic_pixels)`.
  * If `ratio < 0.08` -> **Low**, else if `ratio < 0.22` -> **Moderate**, else -> **High**.
* **UI requirement:** Display this estimated severity as a *"Heuristic Visual Proxy (Not a scientific prediction)"* and let the farmer manually select/override the severity level to dynamically fetch recommendations.

---

## 7. Configuration and Translation Lookup JSON Schema

The `disease_knowledge_base.json` provides all translations and recommendation rules. Use the following Dart lookup structures:

```json
{
  "classes": ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"],
  "min_confidence_threshold": 0.65,
  "min_margin_threshold": 0.15,
  "quality_gate_thresholds": { ... },
  "knowledge_base": {
    "Anthracnose": {
      "name": { "English": "...", "Hausa": "..." },
      "symptoms": { "English": "...", "Hausa": "..." },
      "causes": { "English": "...", "Hausa": "..." },
      "prevention": { "English": "...", "Hausa": "..." },
      "management": { "English": "...", "Hausa": "..." },
      "farmer_guidance": { "English": "...", "Hausa": "..." }
    },
    ...
  },
  "recommendations": {
    "Anthracnose": {
      "Low": { "English": "...", "Hausa": "..." },
      "Moderate": { ... },
      "High": { ... }
    },
    ...
  },
  "retake_recommendations": {
    "low_confidence": { "English": "...", "Hausa": "..." },
    "quality_failed": { ... }
  }
}
```

### Displaying Output:
1. Lookup translated disease name cards from `knowledge_base[disease_name]["name"][language]`.
2. Lookup treatment guide from `recommendations[disease_name][severity][language]`.
3. Lookup retake guides from `retake_recommendations["low_confidence"][language]` if confidence gate fails.
