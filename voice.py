import speechd

tts_d = speechd.SSIPClient('test')
tts_d.set_output_module('rhvoice')
tts_d.set_language('ru')
tts_d.set_rate(50)
tts_d.set_punctuation(speechd.PunctuationMode.SOME)

names_spoken = []


def say_hello(predictions):
    for name, (_, _, _, _) in predictions:
        if name not in names_spoken:
            tts_d.speak(f'Здравствуй, {name}')
            names_spoken.append(name)
