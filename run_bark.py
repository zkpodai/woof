from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

import modal

bark_image = modal.Image \
                    .debian_slim() \
                    .apt_install("git") \
                    .pip_install("git+https://github.com/suno-ai/bark.git") \
                    .run_function(preload_models)

stub = modal.Stub("bark-runner", image = bark_image)

if stub.is_inside():
    from bark import generate_audio


@stub.function(gpu="t4")
def talk(text):
    audio_array = generate_audio(text)
    return audio_array

@stub.local_entrypoint()
def main():
    result = talk.call("[laughs] ZKPs are [clears throat] really nice")
    write_wav("bark_generation.wav", SAMPLE_RATE, result)

