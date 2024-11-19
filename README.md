# G2_Youtube
Analyzing hate messages

# YouTube Hate Speech Detection
This application analyzes YouTube comments to detect hate speech using Natural Language Processing and Machine Learning techniques.

## Overview
This tool helps identify and analyze hate speech in YouTube comments, providing a way to monitor and understand toxic content in social media platforms.

## Features
- YouTube comment extraction from video URLs
- Real-time hate speech detection
- Sentiment analysis of comments
- Statistical visualization of results
- Export results to CSV format

## Technology Stack
- Programming Language: Python 3.8+
- Main Libraries:
  - transformers (Hugging Face)
  - pytorch
  - youtube-api-python
  - pandas
  - streamlit

## Model Information
The application uses a fine-tuned BERT model specifically trained for hate speech detection in multiple languages.
- Base Model: multilingual-BERT
- Model Location: [HuggingFace Hub Link](https://huggingface.co/Dolcevitta/toxic-bert-model/tree/main)
- Training Dataset: [Dataset information]
- Performance Metrics:
  - Accuracy de validación: 0.7950
  - Accuracy de entrenamiento: 0.7950
  - Overfitting: 0.00%
  ### Base Model
  - BERT base uncased

  ### Model Details
  - Task: Binary Classification (Toxic/Non-toxic)
  - Language: English
  - Training Dataset: [Toxic Comments Dataset]()

  ### Model Details
  - **Base Architecture:** BERT (bert-base-uncased)
  - **Type:** Binary classification
  - **Maximum sequence length:** 128 tokens
  - **Language:** English
  - **Classes:**
    - 0: Non-toxic
    - 1: Toxic

  ### Training Hyperparameters
  - Batch size: 16
  - Initial learning rate: 2e-5
  - Epochs: 8 (with early stopping)
  - Dropout rate: 0.2
  - Weight decay: 0.01
  - Optimizer: AdamW
  - Scheduler: Linear with warmup

  ### Limitations
  - Model is trained only for English text
  - Maximum input length: 128 tokens
  - Model may have inherent biases from the training dataset
  - Not optimized for very short or very long texts

  ### Intended Use
  This model is designed for:
  - Automatic comment moderation
  - Toxic content detection in online platforms
  - Toxicity analysis in digital conversations

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-hate-speech-detection.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage
1. Run the Streamlit application:
```bash
streamlit run app.py
```

