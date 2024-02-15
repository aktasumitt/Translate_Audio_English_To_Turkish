# Translate Audio from English to Turkish 

## Introduction:

- In this project, I aimed to translate English Audio to Turkish Audio.
- In this project, I used separate models for English Speech to Text, Text Translation to Turkish, and Turkish Text to Speech, and integrated them all into one place. When run sequentially, we achieved audio-to-audio translation.
-  For Speech to Text, I utilized the OpenAI Whisper model, the Bart Model for translation, and the Bark model for Text to Speech.

#### For details :
 - Whisper : https://openai.com/research/whisper
 - Bark: https://huggingface.co/suno/bark
 - Bart: https://huggingface.co/docs/transformers/model_doc/bart

## Dataset:
- I didnt train the models. I just use for prediction so we have just audio for predict. 

## Train:
- firstly, we use pretrained Whisper Model for English speech to text
- Secondly, we use pretrained Bart Model for translating from English text that is output of Whisper to Turkish 
- After that , we use pretrained Bark Model with the output of Bart for transforming from text to speech.
- I use New York City Introduction audio to transform


## Usage: 
- You can use directly Audio to Audio with Concatinated_Models folder. You may need to set paths in config file acording to your Data paths 
- If you want to use the models separately, you can go to their respective folders and use only what you need from there
- Predictions of Models will save "Prediction" folder 







