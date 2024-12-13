import nemo
import nemo.collections.asr as nemo_asr

model ="/ambernet.nemo"
vad_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model)

def detection(audio_path) :
  
  lang = vad_model.get_label(audio_path)
  
  language_map = ["ru","uz"]

  if lang in language_map:
    return lang
  else:
    return "uz"

audio = "test.wav" 
print(detection(audio))
