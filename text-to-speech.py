import paddlehub as hub
import soundfile as sf

# Load deepvoice3_ljspeech module.
module = hub.Module(name="deepvoice3_ljspeech")

# Predict sentiment label
test_texts = ['Simple as this proposition is, it is necessary to be stated',
              'Parakeet stands for Paddle PARAllel text-to-speech toolkit',
              "今 天 天 气 不 错"]
wavs, sample_rate = module.synthesize(texts=test_texts)
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
