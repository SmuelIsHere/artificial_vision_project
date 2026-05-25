# Artificial Vision Project: Pedestrian Detection & Attribute Recognition

## Project Overview
This repository contains the implementation of an advanced computer vision pipeline focused on **Pedestrian Detection**, **Multi-Object Tracking**, and **Pedestrian Attribute Recognition (PAR)**. The system is capable of accurately locating individuals in a video stream, maintaining their identities across frames, and classifying specific visual attributes for each detected person.

## Key Features & Pipeline Architecture
The project integrates state-of-the-art deep learning models into a unified vision pipeline:

1. **Pedestrian Detection:** Utilizes **YOLO (You Only Look Once)** to achieve real-time, high-accuracy detection of pedestrians/people within the scene.
2. **Multi-Object Tracking:** Integrated with **BoT-SORT** to track detected pedestrians seamlessly, ensuring robust ID retention even during occlusions or complex movements.
3. **Pedestrian Attribute Recognition (PAR):** A custom-trained PAR model is deployed to analyze the cropped images of detected pedestrians. The model performs multi-label classification to recognize key human attributes:
   * **Gender Distinction:** Classifying the gender of the subject.
   * **Headwear Detection:** Identifying the presence of a hat/cap (presenza di cappello).
   * **Backpack Detection:** Detecting whether the person is carrying a backpack.

## Methodology & Training
The PAR network was specifically trained to handle the multi-attribute classification task, optimizing accuracy for fine-grained details like accessories and clothing features, while YOLO and BoT-SORT provide the spatial and temporal backbone for the tracking system.

## Model Weights
The trained weights for the PAR model (`unfr_best_loss.pth`) are ignored by Git due to file size constraints. Please download the weights from [https://www.dropbox.com/scl/fi/txp9vdd9dzjtzvix774mf/unfr_best_loss.pth?rlkey=vw31f5eyejsayikaadeph0lnn&st=0uh0sm68&dl=0] and place them in the `par_models/` directory before running the inference script.

## How to Run

Follow these simple steps to configure and execute the pipeline on your video stream:

<Sequence>
{/* Reason: L'ordine dei passaggi è fondamentale: non puoi analizzare il video prima di aver inserito il file e configurato i parametri, né puoi eseguire video_analysis.py senza aver prima generato la mappatura. */}
  <Step title="Add the video source" subtitle="Step 1">
    Place your target video file inside the `videos/` directory and name it exactly **`video.mp4`**.
  </Step>
  <Step title="Configure the project" subtitle="Step 2">
    Open the **`config.txt`** file and set your desired configuration parameters (such as model thresholds or path settings).
  </Step>
  <Step title="Generate the mapping" subtitle="Step 3">
    Run the mapping script to initialize the coordinates and system layout:
```bash
    python mapping.py
    ```
  </Step>
  <Step title="Execute the pipeline" subtitle="Step 4">
    Run the main analysis script to start pedestrian detection, tracking, and attribute recognition:
```bash
    python video_analysis.py
    ```
  </Step>
  <Step title="Check the outputs" subtitle="Step 5">
    Once processing is complete, the tracking and classification metrics will be saved directly in **`result.txt`**.
  </Step>
</Sequence>
