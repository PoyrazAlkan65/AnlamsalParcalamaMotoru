from TTS.api import TTS
 
# XTTS v2 modeli - çok dilli, Türkçe destekli, ses klonlama yapabiliyor
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
 
ruya_metni = (
    "Rüyanda uçtuğunu görmek, özgürlük arayışını ve hayatındaki "
    "kısıtlamalardan kurtulma isteğini yansıtır. Eğer uçarken korku "
    "hissediyorsan, bu durum kontrolü kaybetme endişesiyle ilişkili olabilir. "
    "Ancak huzurlu bir uçuş, özgüveninin ve iç gücünün güçlendiğine işaret eder."
)
 
# Not: XTTS v2 bir referans ses (speaker_wav) ister.
# Kendi sesinden 6-10 saniyelik temiz bir .wav kaydı al ve yolunu aşağıya yaz.
REFERANS_SES = "data\\inputs\\referans_ses.wav"
 
tts.tts_to_file(
    text=ruya_metni,
    speaker_wav=REFERANS_SES,
    language="tr",
    file_path="data\\outputs\\tts_demo\\ruya_yorumu_coqui3.wav",
)
 
print("Ses dosyası oluşturuldu: ruya_yorumu_coqui3.wav")
 