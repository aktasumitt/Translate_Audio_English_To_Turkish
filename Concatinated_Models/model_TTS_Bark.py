from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import numpy

class Bark():
    
    def __init__(self,Text:str,voice_preset:str,Save_Path:str,SAVE_PREDICT_AUDIO:bool):
        
        self.SAVE_PREDICT_AUDIO=SAVE_PREDICT_AUDIO
        self.Save_Path=Save_Path
        self.Text=Text
        self.voice_preset=voice_preset
    
    
    def Crate_Model(self):
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark")
        return model,processor
    
    
    
    def Training(self,model,processor):
        self.Text[0]=self.Text[0].replace(":",".")
        self.Text[0]=self.Text[0].replace(";",".")

        generated_list=[]
        for text in self.Text[0].split("."):
            inputs = processor(text, voice_preset=self.voice_preset)
            
            audio_array = model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            generated_list.append(audio_array)
        
        return generated_list[:-1]
        
    
    def Save_Audio(self,model,generated_list):
        
        concat_audio = generated_list[0]

        for audio in generated_list[1:]:
            concat_audio = numpy.concatenate(([concat_audio,audio]))
        
        sample_rate = model.generation_config.sample_rate
        write(self.Save_Path, rate=sample_rate, data=concat_audio)
    
    
    
    def forward(self):
        
        model,processor=self.Crate_Model()
        generated_list=self.Training(model=model,processor=processor)
        
        if self.SAVE_PREDICT_AUDIO==True:
            self.Save_Audio(model=model,generated_list=generated_list)
        
        return generated_list