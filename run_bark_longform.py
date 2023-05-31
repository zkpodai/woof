import modal
import numpy as np
from bark.generation import (
    preload_models
)
from bark import SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
import nltk


def install_dependencies():
    preload_models()
    nltk.download('punkt')


bark_image = modal.Image \
    .debian_slim() \
    .apt_install("git") \
    .pip_install("git+https://github.com/suno-ai/bark.git") \
    .pip_install("nltk") \
    .run_function(install_dependencies)

stub = modal.Stub("bark-runner", image=bark_image)

if stub.is_inside():
    from bark import generate_audio

SPEAKER = "v2/en_speaker_6"


@stub.function(gpu="a10g")
def talk_sentence(sentence):
    return generate_audio(sentence, history_prompt=SPEAKER)


@stub.function()
def talk(text):
    sentences = nltk.sent_tokenize(text)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    pieces = []
    for audio_array in talk_sentence.map(sentences):
        pieces += [audio_array, silence.copy()]

    return np.concatenate(pieces)


long_script = """
This is particularly useful for behaviors that happen too infrequently 
to become habitual. Things you have to do monthly or yearly, like
 rebalancing your investment portfolio are never repeated frequently 
 enough to become a habit, so they benefit in particular 
 from technology.
 
Goal setting suffers from a serious case of survivorship bias. We concentrate 
on the people who end up winning—the survivors—and mistakenly assume that 
ambitious goals led to their success while overlooking all of the people 
who had the same objective but didn’t succeed.

Every Olympian wants to win a gold medal. Every candidate wants to get the job.
 And if successful and unsuccessful people share the same goals, then the goal
  cannot be what differentiates the winners from the losers. It wasn’t the 
  goal of winning the Tour de France that propelled the British cyclists 
  to the top of the sport. Presumably, they had wanted to win the race 
  every year before—just like every other professional team. The goal had 
  always been there. It was only when they implemented a system of 
  continuous small improvements that they achieved a different outcome.

Problem #2: Achieving a goal is only a momentary change.
Imagine you have a messy room and you set a goal to clean it. If you summon
 the energy to tidy up, then you will have a clean room—for now. But if 
 you maintain the same sloppy, pack-rat habits that led to a messy room 
 in the first place, soon you’ll be looking at a new pile of clutter 
 and hoping for another burst of motivation. You’re left chasing the same
  outcome because you never changed the system behind it. You treated a
   symptom without addressing the cause.
 
 """.replace(
    "\n", " "
).strip()


@stub.local_entrypoint()
def main():
    result = talk.call(long_script)
    write_wav("longform.wav", SAMPLE_RATE, result)
