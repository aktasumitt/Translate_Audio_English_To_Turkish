import config,model_TTS_Bark

Predicted_Audio=model_TTS_Bark.Bark(Text=config.Text_Bark,
                                    voice_preset=config.voice_preset,
                                    SAVE_PREDICT_AUDIO=config.SAVE_PREDICT_AUDIO,
                                    Save_Path=config.SAVE_PATH
                                    )