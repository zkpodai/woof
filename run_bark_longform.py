import modal
import numpy as np
from bark import SAMPLE_RATE
from bark.generation import (
    preload_models,
)
from scipy.io.wavfile import write as write_wav

bark_image = modal.Image \
    .debian_slim() \
    .apt_install("git") \
    .pip_install("git+https://github.com/suno-ai/bark.git") \
    .pip_install("nltk") \
    .run_function(preload_models)

stub = modal.Stub("bark-runner", image=bark_image)


@stub.function(gpu="a100-20g")
def talk(text):
    from bark import generate_audio, SAMPLE_RATE
    import nltk  # we'll use this to split into sentences
    nltk.download('punkt')

    sentences = nltk.sent_tokenize(text)
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    return np.concatenate(pieces)


long_script = """
Hey, have you heard about this new text-to-audio model called "Bark"? 
Apparently, it's the most realistic and natural-sounding text-to-audio model 
out there right now. People are saying it sounds just like a real person speaking. 
I think it uses advanced machine learning algorithms to analyze and understand the 
nuances of human speech, and then replicates those nuances in its own speech output. 
It's pretty impressive, and I bet it could be used for things like audiobooks or podcasts. 
In fact, I heard that some publishers are already starting to use Bark to create audiobooks. 
It would be like having your own personal voiceover artist. I really think Bark is going to 
be a game-changer in the world of text-to-audio technology.
""".replace("\n", " ").strip()


@stub.local_entrypoint()
def main():
    result = talk.call(long_script)
    write_wav("longform.wav", SAMPLE_RATE, result)
