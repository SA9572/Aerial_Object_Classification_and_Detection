# Aerial Object Classification & Detection

Aerial_Object_Classification_Detection/
│
├── data/
│   ├── classification_dataset/        ← Raw bird vs drone classification images
│   │   ├── train/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   ├── valid/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   └── test/
│   │       ├── bird/
│   │       └── drone/
│   │
│   ├── object_detection_dataset/      ← Raw YOLOv8 detection dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   ├── metadata/                      ← class names, CSV labels, mapping
│   └── processed/                     ← resized, augmented, normalized images
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          ← Dataset inspection & visualization
│   ├── 02_preprocessing_cnn.ipynb         ← Resize/clean/augment classification dataset
│   ├── 03_train_custom_cnn.ipynb          ← Build + train custom CNN model
│   ├── 04_transfer_learning.ipynb         ← ResNet/MobileNet/EfficientNet training
│   ├── 05_model_evaluation.ipynb          ← Confusion matrix, F1-score, graphs
│   ├── 06_yolov8_training.ipynb           ← Train YOLOv8 detection model
│   ├── 07_yolov8_inference.ipynb          ← Run YOLO inference on test images
│   └── 08_streamlit_app_test.ipynb        ← Test real predictions for deployment
│
├── models/
│   ├── classification/
│   │   ├── custom_cnn_best.h5
│   │   ├── transfer_resnet50_best.h5
│   │   ├── transfer_mobilenet_best.h5
│   │   └── history/                       ← Training logs (CSV / pickle)
│   ├── detection/
│   │   ├── yolov8n.pt
│   │   ├── yolov8s.pt
│   │   └── runs/                          ← YOLOv8 training runs folder
│
├── src/
│   ├── data/
│   │   ├── preprocess.py              ← resizing, normalization functions
│   │   ├── augment.py                 ← augmentation pipelines
│   │   └── split_dataset.py           ← auto-split to train/valid/test
│   │
│   ├── models/
│   │   ├── cnn_model.py               ← build CNN architecture
│   │   ├── transfer_model.py          ← ResNet/MobileNet/EfficientNet builders
│   │   ├── train.py                   ← train loop for classification
│   │   ├── evaluate.py                ← confusion matrix / reports
│   │   └── utils.py                   ← early stopping, checkpoints
│   │
│   ├── detection/
│   │   ├── prepare_yolo.py            ← YOLOv8 data.yaml generator
│   │   ├── train_yolo.py              ← YOLOv8 training automation
│   │   └── infer_yolo.py              ← bounding box predictions
│   │
│   ├── deployment/
│   │   ├── predict.py                 ← loading model + predicting image
│   │   ├── utils.py                   ← pre/post processing
│   │   └── classes.json               ← {0: "bird", 1: "drone"}
│   │
│   └── streamlit/
│       └── app.py                     ← final UI for classification/detection
│
├── configs/
│   ├── classification_config.json     ← hyperparameters for CNN or TL
│   ├── yolov8_data.yaml               ← YOLO dataset paths
│   ├── yolov8_config.yaml             ← YOLO training parameters
│   └── env.txt                        ← Required environment variables
│
├── logs/
│   ├── training_logs.csv
│   ├── tensorboard/
│   └── streamlit_app.log
│
├── reports/
│   ├── results.md                     ← Summary of model performance
│   ├── model_comparison.xlsx
│   └── architecture_diagrams/
│
├── streamlit_app/
│   ├── model/                         ← copied best model for deployment
│   ├── assets/                        ← demo images, logo
│   └── app.py
│
├── requirements.txt
├── README.md
└── setup.sh
