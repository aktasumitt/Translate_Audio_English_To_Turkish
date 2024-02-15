from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write

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
        
        inputs = processor(self.Text, voice_preset=self.voice_preset)
        
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array
        
    
    def Save_Audio(self,model,audio_array):
        
        sample_rate = model.generation_config.sample_rate
        write(self.Save_Path, rate=sample_rate, data=audio_array)
    
    
    def forward(self):
        
        model,processor=self.Crate_Model()
        audio_array=self.Training(model=model,processor=processor)
        
        if self.SAVE_PREDICT_AUDIO==True:
            self.Save_Audio(model=model,audio_array=audio_array)
        
        return audio_array
        
        