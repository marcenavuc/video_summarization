# Video summarization project

## Description
Project about how to use LLM and visual transformers to solve video summarization task. To produce importance scores
for each frame from video is used image caption model, that returns description of each image, and automatic speech
recognition to transform audio records from video to text. After that 2 pieces of text have concatenated and put
into LLM model with prompt of summarize text with specific schema. 

## Task definition
Video summarization is the task for set to each frame of video `importance score`, which has value from 0 to 1, where
1 means that frame is important, 0 means that frame is useless and has not valuable information. This task has a lot of
usages starts from cybersecurity to extract importance subvideos from long video stream and ends on crop podcast video
to publish this in advertisement or use in TikTok, YouTube Shorts. 

### Solution performance
This paragraph will contain some performance results of the whole solution.
On this time there are achievement of 51% f1-score on one sample

## Getting started
1. Download dataset from Google Drive 
   https://drive.google.com/drive/folders/1nED17YRACPdQ5Zdrxg6QPGarxTzT7tNY?usp=share_link
   or use following script to download dataset
   ```bash
   pip install gdown && gdown https://drive.google.com/drive/folders/1nED17YRACPdQ5Zdrxg6QPGarxTzT7tNY?usp=share_link
   ```
2. Choose config file or write our own config
3. Run project
```bash
python video_summarization/pipelines/predict_small_sample.py
```
