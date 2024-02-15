import torch,model_ASR,model_Translate,model_TTS_Bark,configs,time

if torch.cuda.is_available():
    devices="cuda:0"


# Start time to control time
start_time=time.time() 



# Asr Model
Result_text_ASR=model_ASR.ASR_Model(devices=devices,
                                    model_type=configs.MODEL_TYPE,
                                    audio_path=configs.Audio_Path_ASR,
                                    save_path=configs.Save_Path_ASR,
                                    SAVE_ASR=configs.SAVE_ASR).forward()



# Translate Model
Result_text_Translated=model_Translate.Translate_Model(devices=devices,
                                                       Text=Result_text_ASR,
                                                       SOURCE_TEXT_LANG=configs.SOURCE_TEXT_LANG,
                                                       TRANSLATED_LAN=configs.TRANSLATED_LAN,
                                                       SAVE_PATH_TRANSLATE=configs.SAVE_PATH_TRANSLATE).forward()

# TTS Model
Predicted_Audio=model_TTS_Bark.Bark(Text=Result_text_Translated, 
                                    voice_preset=configs.voice_preset,
                                    SAVE_PREDICT_AUDIO=configs.SAVE_PREDICT_AUDIO,
                                    Save_Path=configs.SAVE_PATH
                                    ).forward()



#Finish time to control time
finish_time=time.time()


print(f"\n***Training Has Taken {((finish_time-start_time)/60):.3f} Minute***")