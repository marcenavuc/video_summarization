import pandas as pd


def get_text_recognition(asr_model, audio_file):
    asr_result = asr_model.transcribe(audio_file)
    return pd.DataFrame([
        {
            'start_second': segment['start'],
            'end_second': segment['end'],
            'text': segment['text'],
        }
        for segment in asr_result['segments']
    ])
