



# Anomaly Detection in Video Surveillance


## Project Overview
This project implements a deep learning-based anomaly detection system using two distinct architectures: **3D Convolutional Neural Network (3DCNN)** and **a hybrid VGG16+LSTM** networks. 
The system is designed to analyze video sequences and classify them as either *Anomaly* or *Non-anomaly* scenes.

## Model Architectures & Performance
### 1. 3DCNN Model (`anomaly.ipynb`)
The 3DCNN model directly processes spatio-temporal data for effective feature extraction across video frames.

- **Architecture**:
  - Two 3D Convolutional blocks (with 32 and 64 filters)
  - 3D MaxPooling layers
  - Dense layers for final classification

- **Performance**:
  - **Training Accuracy**: 96.57%
  - **Validation Accuracy**: 85.85%
  - **Training Loss**: 0.0961
  - **Validation Loss**: 0.5785

### 2. VGG16+LSTM Model (`anomaly_vgg16+lstm.ipynb`)
The VGG16+LSTM model is a hybrid architecture that combines VGG16 (pretrained on ImageNet) for extracting spatial features from individual frames and LSTM layers for capturing temporal dependencies across frames.

- **Architecture**:
  - VGG16 pretrained model for spatial feature extraction
  - LSTM layers to capture temporal patterns in frame sequences
  - `TimeDistributed` wrapper for handling frames as individual time steps

- **Performance**:
  - **Training Accuracy**: 95.85%
  - **Validation Accuracy**: 91.59%
  - **Training Loss**: 0.1170
  - **Validation Loss**: 0.2251

## Dataset Structure
The dataset includes video clips categorized as Anomaly and Non-Anomaly scenes. Each video is split into frames and organized for training and testing the models.

For the dataset, follow this link: 
[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) , 
[violencedetectionsystem Dataset](https://www.kaggle.com/datasets/kalashkalwani/violencedetectionsystem).

```
dataset/
├── anomaly1/
│   ├── fight/
│   │   └── [fight videos in .mp4 format]
│   └── noFight/
│       └── [non-fight videos in .mp4 format]
├── anomaly2/
│   ├── NonViolence/
│   │   └── [Non-Violence videos in .mp4 format]
│   └── Violence/
│       └── [Violence videos in .mp4 format
└── Frames/
    ├── train/
    │   ├── anomaly/
    │   └── nonanomaly/
    └── test/
        ├── anomaly/
        └── nonanomaly/
```

## Requirements
Install the necessary packages with the following:

```bash
pip install -r requirements.txt
```

**Required Libraries**:
- `tensorflow>=2.0.0`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`

## Installation
1. Clone this repository.
2. Install the required packages listed above.

## Project Files
1. **`anomaly.ipynb`**: Implements the 3DCNN model.
   - Data preprocessing
   - 3DCNN model architecture
   - Training, evaluation, and performance visualization

2. **`anomaly_vgg16+lstm.ipynb`**: Implements the VGG16+LSTM model.
   - Data preprocessing
   - VGG16-based feature extraction
   - Temporal analysis with LSTM
   - Training, evaluation, and performance visualization

## Data Preprocessing
1. Each video is split into frames at 15 frames per second (FPS).
2. Frames are resized to 224x224 pixels to match VGG16’s input requirements.
3. Data is split into 90% training and 10% testing.
4. Frames are normalized by dividing pixel values by 255.0.

## Model Parameters
- **Input Shape**: (5, 224, 224, 3) — 5 frames per sequence
- **Sequence Length**: 5 frames
- **Temporal Stride**: 2
- **Learning Rate**: 0.0001
- **Batch Size**: 10
- **Epochs**: 10

## Project Structure
```
project/
├── dataset/
├── data_files/
│   ├── train/
│   └── test/
├── anomaly.ipynb
├── anomaly_vgg16+lstm.ipynb
├── anomalyModel.keras
├── VGG16+LSTM.keras
└── README.md
```

## Data Generator
The `ActionDataGenerator` class is responsible for:
- Creating frame sequences for each video
- Data augmentation for model robustness
- Batch preparation and real-time data feeding during model training

## Performance Comparison
| Metric             | 3DCNN | VGG16+LSTM |
|-------------------|--------|------------|
| Training Accuracy | 96.57%  | 95.85%      |
| Validation Accuracy| 85.85% | 91.59%      |
| Training Loss     | 0.0961  | 0.117      |
| Validation Loss   | 0.5785  | 0.2251      |

## Training Process
1. Data is loaded in batches using the custom data generator.
2. Each batch contains sequences of frames processed by the models according to their architecture.
3. Early stopping monitors loss to prevent overfitting.

## Future Improvements
1. Implement additional data augmentation techniques.
2. Experiment with longer temporal sequence lengths for improved context.
3. Test different pre-trained models for feature extraction.
4. Implement real-time processing for video feeds.
5. Use cross-validation for more robust evaluation.
6. Experiment with ensemble methods for even better classification performance.

## Contributing
If you’re interested in contributing, feel free to submit issues, ideas, or pull requests!

## License
This project is licensed under the **MIT License**.

## Contact
For questions or suggestions, contact: [Ankit Chouhan](https://www.linkedin.com/in/ankit-chouhan-b41a87206/).
.
