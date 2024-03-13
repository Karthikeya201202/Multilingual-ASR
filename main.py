import openai
from datasets import load_dataset
from openai import OpenAI
from transformers import pipeline
openai.api_key="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"
client = OpenAI(api_key="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")
audio_file1 = open("harvard.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file1, 
  response_format="text" #text format. we can get the output in several formats.
)
txt=transcription.text
print(txt)  #this produces the transcribes of the audio.

#this is the translation of the audio file in desired language
audio_file2= open("hindi.wav", "rb")
translation = client.audio.translations.create(
  model="whisper-1", 
  file=audio_file2,
   #hindi to english translation
)
print(translation.text)


# Summarization task: summarizing the transcribed text 
# using facebook bart model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer(txt, max_length=130, min_length=30, do_sample=False))
