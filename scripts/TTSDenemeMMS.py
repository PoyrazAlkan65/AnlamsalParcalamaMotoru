from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

ruya_metni = "Rüyanda gördüğün deniz, bilinçaltındaki duygusal derinliği simgeler. Dalgaların sakinliği, iç huzurunu işaret eder."

inputs = tokenizer(ruya_metni, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write(
    "data/outputs/tts_demo/ruya_yorumu_mms.wav",
    rate=model.config.sampling_rate,
    data=output.numpy().squeeze()
)

print("Tamamlandı: ruya_yorumu_mms.wav")