# Project Structure Documentation

## Aerial Object Classification & Detection

---

## 📁 Complete Project Directory Tree

Aerial Object Classification & Detection/
│
├── 📄 Project Title.pdf                              [EXTERNAL]
├── 📄 Project Title - Recreated.md                   [AUTO]
├── 📄 README_ProjectTitle.md                         [AUTO]
├── 📄 STRUCTURE.md                                   [AUTO] ← You are here
│
├── 📁 dataset/                                        [USER PROVIDED]
│   │
│   ├── 📁 Classification Dataset/
│   │   │
│   │   ├── 📁 train-20251116T141539Z-1-001/
│   │   │   └── 📁 train/
│   │   │       ├── 📁 bird/                         1,414 images
│   │   │       └── 📁 drone/                        1,248 images
│   │   │
│   │   ├── 📁 valid-20251116T141416Z-1-001/
│   │   │   └── 📁 valid/
│   │   │       ├── 📁 bird/                           217 images
│   │   │       └── 📁 drone/                          225 images
│   │   │
│   │   └── 📁 test-20251116T141755Z-1-001/
│   │       └── 📁 test/
│   │           ├── 📁 bird/                           121 images
│   │           └── 📁 drone/                           94 images
│   │
│   └── 📁 Object Detection Dataset (YOLOv8 Format)/
│       │
│       ├── 📄 data.yaml                              [USER PROVIDED]
│       ├── 📄 README.dataset.txt                     [USER PROVIDED]
│       ├── 📄 README.roboflow.txt                    [USER PROVIDED]
│       │
│       ├── 📁 train-20251116T144736Z-1-001/
│       │   └── 📁 train/
│       │       ├── 📁 images/                        2,662 images
│       │       └── 📁 labels/                        2,662 .txt files (YOLO format)
│       │
│       ├── 📁 valid-20251116T144000Z-1-001/
│       │   └── 📁 valid/
│       │       ├── 📁 images/                          442 images
│       │       └── 📁 labels/                          442 .txt files
│       │
│       └── 📁 test-20251116T144001Z-1-001/
│           └── 📁 test/
│               ├── 📁 images/                          215 images
│               └── 📁 labels/                          215 .txt files
│
├── 📁 notebooks/                                      [AUTO FOLDER]
│   ├── 01_data_exploration.ipynb                     [TO CREATE]
│   ├── 02_custom_cnn_training.ipynb                  [TO CREATE]
│   ├── 03_transfer_learning_training.ipynb           [TO CREATE]
│   ├── 04_model_evaluation.ipynb                     [TO CREATE]
│   ├── 05_yolov8_training.ipynb                      [TO CREATE]
│   └── 06_final_inference.ipynb                      [TO CREATE]
│
├── 📁 scripts/                                        [AUTO FOLDER]
│   ├── 📄 data_preprocessing.py                      [AUTO]
│   ├── 📄 train_classification.py                    [AUTO]
│   ├── 📄 train_yolov8.py                            [AUTO]
│   ├── 📄 app.py                                     [AUTO - Streamlit]
│   ├── 📄 evaluate_models.py                         [TO CREATE]
│   ├── 📄 inference.py                               [TO CREATE]
│   └── 📄 utils.py                                   [TO CREATE]
│
├── 📁 models/                                         [AUTO FOLDER]
│   ├── 📁 classification/
│   │   ├── custom_cnn.h5                             [TO CREATE]
│   │   ├── resnet50_finetuned.h5                     [TO CREATE]
│   │   ├── mobilenet_finetuned.h5                    [TO CREATE]
│   │   └── efficientnetb0_finetuned.h5               [TO CREATE]
│   │
│   └── 📁 detection/
│       └── yolov8_best.pt                            [TO CREATE]
│
├── 📁 outputs/                                        [AUTO FOLDER]
│   ├── 📁 training_logs/
│   │   ├── custom_cnn_history.json                   [TO CREATE]
│   │   ├── resnet50_history.json                     [TO CREATE]
│   │   └── mobilenet_history.json                    [TO CREATE]
│   │
│   ├── 📁 confusion_matrices/
│   │   ├── custom_cnn_cm.png                         [TO CREATE]
│   │   ├── resnet50_cm.png                           [TO CREATE]
│   │   └── classification_report.txt                 [TO CREATE]
│   │
│   ├── 📁 yolov8_training/
│   │   └── bird_drone_detection/                     [TO CREATE]
│   │       ├── 📁 weights/
│   │       │   ├── best.pt                           [TO CREATE]
│   │       │   └── last.pt                           [TO CREATE]
│   │       ├── 📁 results/
│   │       │   ├── confusion_matrix.png              [TO CREATE]
│   │       │   ├── F1_curve.png                      [TO CREATE]
│   │       │   └── results.csv                       [TO CREATE]
│   │       └── 📁 predictions/                       [TO CREATE]
│   │
│   └── 📁 test_predictions/
│       ├── classification_results.json               [TO CREATE]
│       └── detection_results.json                    [TO CREATE]
│
├── 📁 data_exploration/                              [AUTO FOLDER]
│   ├── 📄 class_distribution.png                     [TO CREATE]
│   ├── 📄 sample_images.png                          [TO CREATE]
│   ├── 📄 image_size_analysis.png                    [TO CREATE]
│   └── 📄 dataset_statistics.json                    [TO CREATE]
│
├── 📁 config/                                         [AUTO FOLDER]
│   ├── 📄 data.yaml                                  [AUTO - YOLOv8 config]
│   ├── 📄 training_config.yaml                       [TO CREATE]
│   └── 📄 model_config.yaml                          [TO CREATE]
│
├── 📁 .github/                                        [TO CREATE - Optional]
│   └── 📁 workflows/
│       └── 📄 ci_cd.yml                              [TO CREATE]
│
└── 📄 requirements.txt                               [AUTO]

---

## 📋 File-by-File Documentation

### 🔵 Root Level Files

| File | Origin | Created By | Purpose | Status |
|------|--------|-----------|---------|--------|
| `Project Title.pdf` | Uploaded | User | Original project brief | Reference |
| `Project Title - Recreated.md` | Generated | Auto | Markdown version of project brief | ✅ Ready |
| `README_ProjectTitle.md` | Generated | Auto | Quick start guide | ✅ Ready |
| `STRUCTURE.md` | Generated | Auto | This file - complete project structure | ✅ Ready |
| `requirements.txt` | Generated | Auto | Python dependencies (numpy, tensorflow, etc.) | ✅ Ready |

### 🔴 Dataset Files (User Provided)

**Classification Dataset:** `dataset/Classification Dataset/`

- **Train**: 1,414 birds + 1,248 drones = **2,662 total**
- **Validation**: 217 birds + 225 drones = **442 total**
- **Test**: 121 birds + 94 drones = **215 total**
- **Format**: .jpg RGB images
- **Status**: ✅ Ready to use

**Object Detection Dataset (YOLOv8):** `dataset/Object Detection Dataset (YOLOv8 Format)/`

- **Train**: 2,662 images + 2,662 .txt annotation files
- **Validation**: 442 images + 442 .txt annotation files
- **Test**: 215 images + 215 .txt annotation files
- **Format**: YOLO format (normalized coordinates in .txt files)
- **Status**: ✅ Ready to use

### 🟢 Scripts (Auto-Generated, Ready for Implementation)

| Script | Purpose | Status | To Run |
|--------|---------|--------|--------|
| `scripts/data_preprocessing.py` | Load images, normalize, resize | ✅ Skeleton ready | `python scripts/data_preprocessing.py` |
| `scripts/train_classification.py` | Train Custom CNN & Transfer Learning models | ✅ Skeleton ready | `python scripts/train_classification.py` |
| `scripts/train_yolov8.py` | Train YOLOv8 detection model | ✅ Skeleton ready | `python scripts/train_yolov8.py` |
| `scripts/app.py` | Streamlit web interface | ✅ Skeleton ready | `streamlit run scripts/app.py` |
| `scripts/evaluate_models.py` | Model evaluation & comparison | ⏳ To create | - |
| `scripts/inference.py` | Run inference on new images | ⏳ To create | - |
| `scripts/utils.py` | Helper functions | ⏳ To create | - |

### 📔 Notebooks (To Create)

| Notebook | Purpose | Status |
|----------|---------|--------|
| `notebooks/01_data_exploration.ipynb` | Visualize dataset, class distribution, sample images | ⏳ To create |
| `notebooks/02_custom_cnn_training.ipynb` | Train custom CNN from scratch | ⏳ To create |
| `notebooks/03_transfer_learning_training.ipynb` | Fine-tune ResNet50, MobileNet, EfficientNetB0 | ⏳ To create |
| `notebooks/04_model_evaluation.ipynb` | Confusion matrix, classification report, comparisons | ⏳ To create |
| `notebooks/05_yolov8_training.ipynb` | YOLOv8 training and validation | ⏳ To create |
| `notebooks/06_final_inference.ipynb` | Test predictions on test set | ⏳ To create |

### 📁 Directories Breakdown

#### `models/` - Trained Model Storage

**Purpose**: Store serialized trained models

- `classification/`: Classification models (.h5 or .keras format)

- `detection/`: YOLOv8 model weights (.pt format)
- **When populated**: After successful training
- **Output format**: HDF5 (.h5) for Keras models, PyTorch (.pt) for YOLOv8

#### `outputs/` - Training Results & Predictions

**Purpose**: Store training logs, confusion matrices, predictions

- `training_logs/`: JSON files with training history
- `confusion_matrices/`: PNG images of confusion matrices
- `yolov8_training/`: YOLOv8 training artifacts (auto-generated during training)
- `test_predictions/`: Inference results on test set
- **When populated**: During and after model training

#### `data_exploration/` - EDA Artifacts

**Purpose**: Store data exploration visualizations

- Class distribution histograms
- Sample image grids
- Dataset statistics JSON
- **When populated**: During initial data exploration phase

#### `config/` - Configuration Files

**Purpose**: Store configuration for models and training

- `data.yaml`: YOLOv8 dataset configuration (auto-generated)
- `training_config.yaml`: Training hyperparameters (to create)
- `model_config.yaml`: Model architecture settings (to create)

#### `notebooks/` - Jupyter Notebooks

**Purpose**: Interactive development and experimentation

- All notebooks are to be created
- Should follow workflow steps in project brief

---

## 🎯 Workflow & Expected Outputs

### Phase 1: Data Exploration ✅ Ready

**Files**: `notebooks/01_data_exploration.ipynb`

**Outputs**:

- Class distribution visualization → `data_exploration/class_distribution.png`
- Sample images grid → `data_exploration/sample_images.png`
- Dataset statistics → `data_exploration/dataset_statistics.json`

### Phase 2: Model Training (Classification)

**Files**:

- `scripts/data_preprocessing.py` (run first to prepare data)
- `scripts/train_classification.py`
- `notebooks/02_custom_cnn_training.ipynb` & `03_transfer_learning_training.ipynb`

**Outputs**:

- Trained models → `models/classification/`
- Training history → `outputs/training_logs/`
- Confusion matrices → `outputs/confusion_matrices/`

**Expected output example**:

Training Custom CNN...
Epoch 1/100 - Loss: 0.654, Accuracy: 0.612
Epoch 2/100 - Loss: 0.489, Accuracy: 0.754
...
Model saved to: models/classification/custom_cnn.h5
Accuracy: 87.3%  Precision: 89.2%  Recall: 85.1%

### Phase 3: Object Detection Training (YOLOv8)

**Files**:

- `scripts/train_yolov8.py`
- `notebooks/05_yolov8_training.ipynb`

**Outputs**:

- Model weights → `models/detection/yolov8_best.pt`
- Training results → `outputs/yolov8_training/`
- Validation plots → `outputs/yolov8_training/bird_drone_detection/results/`

**Expected output example**:

YOLOv8 Training Results:
Epoch [1/100] - Loss: 0.845, mAP50: 0.654
Epoch [100/100] - Loss: 0.234, mAP50: 0.926
Model saved to: models/detection/yolov8_best.pt
Validation mAP50: 92.6%

### Phase 4: Model Evaluation

**Files**:

- `scripts/evaluate_models.py`
- `notebooks/04_model_evaluation.ipynb`

**Outputs**:

- Classification report with metrics
- Model comparison table
- Confusion matrices for each model
- Test set predictions

### Phase 5: Deployment

**Files**: `scripts/app.py`

**To run**:

```powershell
streamlit run scripts/app.py
```

**Expected output**:

- Web UI at `http://localhost:8501`
- Upload image functionality
- Real-time predictions with confidence scores
- Optional bounding box visualizations from YOLOv8

---

## 🔑 Key File Statuses

### ✅ Ready to Use Now

- `dataset/` - All datasets provided
- `Project Title - Recreated.md` - Project brief
- `requirements.txt` - Dependencies
- `scripts/data_preprocessing.py` - Skeleton ready
- `scripts/train_classification.py` - Skeleton ready
- `scripts/train_yolov8.py` - Skeleton ready
- `scripts/app.py` - Streamlit skeleton
- `config/data.yaml` - YOLOv8 config template

### ⏳ To Create Next

- Notebooks (01-06) - Data exploration, training, evaluation
- `scripts/evaluate_models.py` - Model evaluation script
- `scripts/inference.py` - Inference helper
- `scripts/utils.py` - Utility functions
- Training configuration files

### 🔄 Auto-Generated During Training

- `models/classification/*.h5` - Trained model files
- `models/detection/yolov8_best.pt` - Best YOLOv8 weights
- `outputs/training_logs/*.json` - Training history
- `outputs/confusion_matrices/*.png` - Evaluation plots
- `outputs/yolov8_training/` - YOLOv8 complete training artifacts

---

## 🚀 Quick Start Commands

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data exploration
# jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Run training
# python scripts/train_classification.py
# python scripts/train_yolov8.py

# 4. Launch Streamlit app (after models trained)
streamlit run scripts/app.py
```

---

## 📊 Summary Statistics

| Category | Count | Status |
|----------|-------|--------|
| **Datasets** | 2 | ✅ Ready |
| **Dataset images** | 3,319 | ✅ Ready |
| **Scripts (skeleton)** | 4 | ✅ Ready |
| **Notebooks** | 6 | ⏳ To create |
| **Folders** | 9 | ✅ Created |
| **Configuration files** | 1 | ✅ Ready |
| **Total auto-generated files** | 11 | ✅ Complete |

---

## 📝 Notes

1. **Data Location**: All datasets are in `dataset/` folder and ready to use
2. **Model Checkpoints**: After training, models will be saved in `models/` with .h5 (Keras) or .pt (YOLOv8) extensions
3. **Results**: All training outputs (logs, plots, metrics) go to `outputs/`
4. **Reproducibility**: Keep `config/` files versioned for experiment tracking
5. **GPU Training**: Update `scripts/train_yolov8.py` to use GPU device (default: 0)

---

**Last Updated**: November 17, 2025
**Project Status**: Structure Ready | Implementation In Progress
