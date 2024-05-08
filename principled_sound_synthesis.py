import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import librosa
import numpy as np
from numpy import exp
from scipy.io.wavfile import write
import math
import matplotlib.pyplot as plt
import pygame
import os
import json

sr = 44100 #sample rate (samples per second)
notefrequencies = {index+21: 2**((d-69)/12)*440 for index, d in enumerate(range(21,89))}
def load_instrument():
    file = filedialog.askopenfile(filetypes=[('JSON files', '*.json')])
    text = file.read()
    file.close()

    [volume, master_pitch, duration, master_base_frequency, evens, odds, free_bar, clamped_bar, rectangular_membrane, ideal_timpani, realistic_timpani, shimmeryness, initial_noisiness, harmonic_unsteadyness, loudness_variation, comparative_rate, decay_rate, clarification_time, clarification_duration, comeup_duration, comedown_duration] = json.loads(text)

    volume_value.set(volume*100.0)
    pitch_value.set(master_pitch)
    duration_value.set(duration)
    base_frequency_value.set(master_base_frequency)
    evens_value.set(evens*100.0)
    odds_value.set(odds*100.0)
    free_bar_value.set(free_bar*100.0)
    clamped_bar_value.set(clamped_bar*100.0)
    rectangular_membrane_value.set(rectangular_membrane*100.0)
    ideal_timpani_value.set(ideal_timpani*100.0)
    realistic_timpani_value.set(realistic_timpani*100.0)
    shimmeryness_value.set(shimmeryness*100.0)
    initial_noisiness_value.set(initial_noisiness*100.0)
    harmonic_unsteadyness_value.set(harmonic_unsteadyness*100.0)
    loudness_variation_value.set(loudness_variation*100.0)
    comparative_rate_value.set(comparative_rate*100.0)
    decay_rate_value.set(decay_rate)
    clarification_time_value.set(clarification_time)
    clarification_duration_value.set(clarification_duration)
    comeup_duration_value.set(comeup_duration)
    comedown_duration_value.set(comedown_duration)

    sliders_update(0)

def generate_random_sequence(length):
    #TODO: Improve random algorithm, possibly specialized ones for each purpose. Compounded sine waves might be a good idea.
    length = int(length)
    l = 150
    h = 500
    n_fft = 2048#*128
    l = int(length/n_fft)
    h = int(n_fft/2+1)

    spec = np.random.rand(h, l)
    spec[int(h/10):] = [0] * l
    inv = librosa.istft(spec)
    repeated = np.tile(inv, int(np.ceil(length / len(inv))))
    out = repeated[:length] / np.max(np.abs(repeated))
    return out

def planck_distribution(pitch, v, volume):
    c = pitch/2.82143937
    a = volume/(pitch**3/exp(pitch/c)-1)
    amplitude = a*(v**3)/(exp(v/c)-1)
    return amplitude

def generate_waveform(volume, pitch, duration, base_frequency, evens, odds, free_bar, clamped_bar, rectangular_membrane, ideal_timpani, realistic_timpani, shimmeryness, initial_noisiness, harmonic_unsteadyness, loudness_variation, comparative_rate, decay_rate, clarification_time, clarification_duration, comeup_duration, comedown_duration):
    secs = duration
    n_fft = 2048*128
    l = int((sr/n_fft)*secs)
    h = int(n_fft/2+1)
    np.random.seed(42)
    times = np.linspace(0, secs, int(sr * secs), endpoint=False)
    output = times * 0

    harmonic_frequencies = {}
    distribution_function = planck_distribution

    #compute overtones
    if evens > 0:
        v = base_frequency
        while True:
            amplitude = evens * distribution_function(pitch, v, volume)
            if amplitude > 0.000000001:
                harmonic_frequencies[v] = amplitude
            else:
                break
            v += base_frequency * 2
    if odds > 0:
        v = base_frequency * 2
        while True:
            amplitude = odds * distribution_function(pitch, v, volume)
            if amplitude > 0.000000001:
                harmonic_frequencies[v] = amplitude
            else:
                break
            v += base_frequency * 2
    if free_bar > 0:
        v = base_frequency
        i = 1
        while True:
            v = 0.441*(i+0.5)**2 * base_frequency
            amplitude = free_bar * distribution_function(pitch, v, volume)
            if amplitude > 0.000000001:
                harmonic_frequencies[v] = amplitude
            else:
                break
            i += 1
    if clamped_bar > 0:
        v = base_frequency
        i = 1
        while True:
            v = 2.81*(i-0.5)**2 * base_frequency
            amplitude = clamped_bar * distribution_function(pitch, v, volume)
            if amplitude > 0.000000001:
                harmonic_frequencies[v] = amplitude
            else:
                break
            i += 1
    if rectangular_membrane > 0:
        b = base_frequency
        harmonics = list(np.array([1, 1.41, 1.73, 2, 2.38, 2.71, 3, 3.37]) * b)
        for v in harmonics:
            amplitude = rectangular_membrane * distribution_function(pitch, v, volume)
            harmonic_frequencies[v] = amplitude
    if ideal_timpani > 0:
        b = base_frequency
        theoretical_harmonics = [1, 1.35, 1.67, 1.99, 2.3, 2.61]
        harmonics = list(np.array(theoretical_harmonics) * b)
        for v in harmonics:
            amplitude = ideal_timpani * distribution_function(pitch, v, volume)
            harmonic_frequencies[v] = amplitude
    if realistic_timpani > 0:
        b = base_frequency
        actual_harmonics = [1, 1.504, 1.742, 2, 2.245, 2.494, 2.8, 2.852, 2.979, 3.462]
        harmonics = list(np.array(actual_harmonics) * b)
        for v in harmonics:
            amplitude = realistic_timpani * distribution_function(pitch, v, volume)
            harmonic_frequencies[v] = amplitude
    '''
    if random_harmonic > 0:
        for i in range(10):
            v = random.randrange(20, 20000)
            amplitude = random_harmonic * a*v**3/(exp(v/c)-1)
            harmonic_frequencies[v] = amplitude
            '''

    if shimmeryness > 0:
        bfreq_random_walk = generate_random_sequence(sr * secs)
    else:
        bfreq_random_walk = np.zeros(int(sr*secs))
    if loudness_variation > 0:
        loudness_random_walk = generate_random_sequence(sr * secs)
    else:
        loudness_random_walk = np.zeros(int(sr*secs))
    for f in harmonic_frequencies:
        if harmonic_unsteadyness > 0:
            harmonic_random_walk = generate_random_sequence(sr * secs)
        else:
            harmonic_random_walk = np.zeros(int(sr*secs))
        output += harmonic_frequencies[f] * np.sin(2 * np.pi * (f + harmonic_random_walk * harmonic_unsteadyness * f / 1000 + bfreq_random_walk * shimmeryness * f / 1000) * times) * exp(decay_rate*times/(comparative_rate**((f-base_frequency)/base_frequency))) + loudness_random_walk * loudness_variation * f / 100000

    if initial_noisiness > 0:
        random_noise = generate_random_sequence(sr * secs)
        i = np.arange(len(output))
        t = np.clip((i - clarification_time) / clarification_duration, 0.0, 1.0)
        smoothstep = t * t * (3 - 2 * t)
        output = initial_noisiness * (random_noise * (1 - smoothstep) + output * smoothstep) + output * (1 - initial_noisiness)
    if comeup_duration > 0:
        i = np.arange(len(output))
        t = np.clip(i / comeup_duration, 0.0, 1.0)
        smoothstep = t * t * (3 - 2 * t)
        output *= smoothstep
    if comedown_duration > 0:
        i = len(output) - np.arange(len(output))
        t = np.clip(i / comedown_duration, 0.0, 1.0)
        smoothstep = t * t * (3 - 2 * t)
        output *= smoothstep

    output *= volume / np.max(output)

    return output, harmonic_frequencies

def generate_audio(mode="single", outfile="generated_audio"):
    # Get slider values
    volume = volume_value.get() / 100.0
    master_pitch = pitch_value.get()
    duration = duration_value.get()
    master_base_frequency = base_frequency_value.get()
    evens = evens_value.get() / 100.0
    odds = odds_value.get() / 100.0
    free_bar = free_bar_value.get() / 100.0
    clamped_bar = clamped_bar_value.get() / 100.0
    rectangular_membrane = rectangular_membrane_value.get() / 100.0
    ideal_timpani = ideal_timpani_value.get() / 100.0
    realistic_timpani = realistic_timpani_value.get() / 100.0
    shimmeryness = shimmeryness_value.get() / 100.0
    initial_noisiness = initial_noisiness_value.get() / 100.0
    harmonic_unsteadyness = harmonic_unsteadyness_value.get() / 100.0
    loudness_variation = loudness_variation_value.get() / 100.0
    comparative_rate = comparative_rate_value.get() / 100.0
    decay_rate = decay_rate_value.get()
    clarification_time = clarification_time_value.get()
    clarification_duration = clarification_duration_value.get()
    comeup_duration = comeup_duration_value.get()
    comedown_duration = comedown_duration_value.get()


    #generate frequencies at which to generate note files
    notes_to_generate = {}
    text = ''
    if mode == "sfz":
        notes_to_generate = notefrequencies
        text = '<group>\n\n'
    else:
        closestmidinote = min(notefrequencies, key=lambda x: abs(notefrequencies[x] - master_base_frequency))
        notes_to_generate = {closestmidinote: master_base_frequency,}

    for midi_note in notes_to_generate:
        base_frequency = notes_to_generate[midi_note]
        pitch = master_pitch * base_frequency / master_base_frequency

        print(f'{midi_note=}, {pitch=}, {base_frequency=}')
        waveform, harmonic_frequencies = generate_waveform(volume, pitch, duration, base_frequency, evens, odds, free_bar, clamped_bar, rectangular_membrane, ideal_timpani, realistic_timpani, shimmeryness, initial_noisiness, harmonic_unsteadyness, loudness_variation, comparative_rate, decay_rate, clarification_time, clarification_duration, comeup_duration, comedown_duration)

        if mode == 'single' or mode == 'graph':
            # Save the audio file using scipy
            audio_filename = outfile + ".wav"
            #write(audio_filename, sr, waveform)
            amplitude = np.iinfo(np.int32).max
            data = amplitude * waveform
            write(audio_filename, sr, data.astype(np.int32))

            if mode == 'graph':
                frequencies = list(harmonic_frequencies.keys())
                amplitudes = list(harmonic_frequencies.values())

                print(frequencies)
                print(amplitudes)
                # Create a bar graph
                plt.bar(frequencies, amplitudes, align='center', width=100, color='black')
                #plt.xscale('log')
                plt.xlim(20, 20000)

                # Set labels and title
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                plt.title('Harmonic Frequencies')

                # Display the graph
                plt.show()

            return audio_filename
        elif mode == 'sfz':
            audio_filename = outfile + "_" + str(midi_note) + ".wav"
            #write(audio_filename, sr, waveform)
            amplitude = np.iinfo(np.int32).max
            data = amplitude * waveform
            write(audio_filename, sr, data.astype(np.int32))
            trunc_filename = audio_filename.split('/')[-1].split('\\')[-1]
            text += f'<region>\nsample={trunc_filename} key={midi_note}\n'
            text += 'loop_mode=no_loop\nloop_start=0\nloop_end=0\n'

    if mode == 'sfz':
        f = open(outfile + '.sfz', 'w')
        f.write(text)
        f.close()
        f = open(outfile + '.json', 'w')
        pcssi = json.dumps([volume, master_pitch, duration, master_base_frequency, evens, odds, free_bar, clamped_bar, rectangular_membrane, ideal_timpani, realistic_timpani, shimmeryness, initial_noisiness, harmonic_unsteadyness, loudness_variation, comparative_rate, decay_rate, clarification_time, clarification_duration, comeup_duration, comedown_duration])
        f.write(pcssi)
        f.close()

def generate_and_play_audio():
    audio_filename = generate_audio()

    # Play the audio file using pygame
    pygame.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()

def graph_audio():
    audio_filename = generate_audio('graph')

    # Play the audio file using pygame
    pygame.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()

def generate_sfz():
    path = filedialog.asksaveasfilename()
    if path:
        audio_filename = generate_audio("sfz", path)
        subprocess.run(['polyphone', '-1', '-i', path + '.sfz'])

def sliders_update(value):
    volume_slider_label.config(text=f"{volume_value.get()}")
    pitch_slider_label.config(text=f"{pitch_value.get()}")
    duration_slider_label.config(text=f"{duration_value.get()}")
    base_frequency_slider_label.config(text=f"{base_frequency_value.get()}")
    evens_slider_label.config(text=f"{evens_value.get()}")
    odds_slider_label.config(text=f"{odds_value.get()}")
    free_bar_slider_label.config(text=f"{free_bar_value.get()}")
    clamped_bar_slider_label.config(text=f"{clamped_bar_value.get()}")
    rectangular_membrane_slider_label.config(text=f"{rectangular_membrane_value.get()}")
    ideal_timpani_slider_label.config(text=f"{ideal_timpani_value.get()}")
    realistic_timpani_slider_label.config(text=f"{realistic_timpani_value.get()}")
    shimmeryness_slider_label.config(text=f"{shimmeryness_value.get()}")
    initial_noisiness_slider_label.config(text=f"{initial_noisiness_value.get()}")
    harmonic_unsteadyness_slider_label.config(text=f"{harmonic_unsteadyness_value.get()}")
    loudness_variation_slider_label.config(text=f"{loudness_variation_value.get()}")
    comparative_rate_slider_label.config(text=f"{comparative_rate_value.get()}")
    decay_rate_slider_label.config(text=f"{decay_rate_value.get()}")
    clarification_time_slider_label.config(text=f"{clarification_time_value.get()}")
    clarification_duration_slider_label.config(text=f"{clarification_duration_value.get()}")
    comeup_duration_slider_label.config(text=f"{comeup_duration_value.get()}")
    comedown_duration_slider_label.config(text=f"{comedown_duration_value.get()}")

root = tk.Tk()
root.title("Principled Sound Synthesis")

# Variables for sliders
volume_value = tk.DoubleVar(value=50)
pitch_value = tk.DoubleVar(value=200)
duration_value = tk.DoubleVar(value=1)
base_frequency_value = tk.DoubleVar(value=200)
evens_value = tk.DoubleVar(value=100)
odds_value = tk.DoubleVar(value=100)
free_bar_value = tk.DoubleVar(value=0)
clamped_bar_value = tk.DoubleVar(value=0)
rectangular_membrane_value = tk.DoubleVar(value=0)
ideal_timpani_value = tk.DoubleVar(value=0)
realistic_timpani_value = tk.DoubleVar(value=0)
#random_harmonic_value = tk.DoubleVar(value=0)
shimmeryness_value = tk.DoubleVar(value=0)
initial_noisiness_value = tk.DoubleVar(value=0)
harmonic_unsteadyness_value = tk.DoubleVar(value=0)
loudness_variation_value = tk.DoubleVar(value=0)
comparative_rate_value = tk.DoubleVar(value=92)
decay_rate_value = tk.DoubleVar(value=-2)
clarification_time_value = tk.DoubleVar(value=0)
clarification_duration_value = tk.DoubleVar(value=0)
comeup_duration_value = tk.DoubleVar(value=1000)
comedown_duration_value = tk.DoubleVar(value=1000)

        # Create sliders
volume_label = ttk.Label(root, text="volume")
pitch_label = ttk.Label(root, text="loudest frequency for test note (Hz)")
duration_label = ttk.Label(root, text="note duration (seconds)")
base_frequency_label = ttk.Label(root, text="base (lowest) frequency for test sample (Hz)")
host_label = ttk.Label(root, text="Harmonic/overtone series type weightings (percent of full volume):")
evens_label = ttk.Label(root, text="even harmonics (strings and open cylinders vibrate on both odd and even harmonics roughly equally, but bassoons generally have a bias towards even harmonics)")
odds_label = ttk.Label(root, text="odd harmonics (one-end closed cylinders vibrate only on odd harmonics, and similar things like clarinets vibrate mostly on odd harmonics)")
free_bar_label = ttk.Label(root, text="ideal free bar harmonics (think xylophones)")
clamped_bar_label = ttk.Label(root, text="ideal clamped bar harmonics (think marimba)")
rectangular_membrane_label = ttk.Label(root, text="ideal rectangular membrane harmonics (golden ratio proportion)")
ideal_timpani_label = ttk.Label(root, text="ideal timpani harmonics")
realistic_timpani_label = ttk.Label(root, text="air atmosphere timpani harmonics")
#random_harmonic_label = ttk.Label(root, text="random harmonics (different than noise)")
shimmeryness_label = ttk.Label(root, text="shimmeryness (how self-consistent the base and overtone frequencies are)")
initial_noisiness_label = ttk.Label(root, text="initial noise/noisiness as a percent (100% is common in percussion instruments but 0% is common for most other things)")
harmonic_unsteadyness_label = ttk.Label(root, text="individual harmonic unsteadyness (how much frequencies increase/decrease on account of random noise)")
loudness_variation_label = ttk.Label(root, text="loudness variation (how much the entire sound increases/decreases randomly)")
comparative_rate_label = ttk.Label(root, text="rate of loss of higher frequency harmonics compared to lower frequency harmonics with note volume decay (a frequency may have 50% the life of the one 100Hz below it)")
decay_rate_label = ttk.Label(root, text="rate of exponential decay of the sound overall")
clarification_time_label = ttk.Label(root, text="amount of time spent as noise before clarifying into note harmonic frequencies (for free bells and blocks and things)")
clarification_duration_label = ttk.Label(root, text="amount of time spent transitioning between noise into note harmonic frequencies")
comeup_duration_label = ttk.Label(root, text="amount of time the note takes to get to peak volume")
comedown_duration_label = ttk.Label(root, text="amount of time the note takes to get to silence once it stops being played")

slider_length = 600

volume_slider = ttk.Scale(root, from_=0, to=100, variable=volume_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
pitch_slider = ttk.Scale(root, from_=20, to=3000, variable=pitch_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
duration_slider = ttk.Scale(root, from_=0, to=10, variable=duration_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
base_frequency_slider = ttk.Scale(root, from_=20, to=3000, variable=base_frequency_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
evens_slider = ttk.Scale(root, from_=0, to=100, variable=evens_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
odds_slider = ttk.Scale(root, from_=0, to=100, variable=odds_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
free_bar_slider = ttk.Scale(root, from_=0, to=100, variable=free_bar_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
clamped_bar_slider = ttk.Scale(root, from_=0, to=100, variable=clamped_bar_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
rectangular_membrane_slider = ttk.Scale(root, from_=0, to=100, variable=rectangular_membrane_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
ideal_timpani_slider = ttk.Scale(root, from_=0, to=100, variable=ideal_timpani_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
realistic_timpani_slider = ttk.Scale(root, from_=0, to=100, variable=realistic_timpani_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
#random_harmonic_slider = ttk.Scale(root, from_=0, to=100, variable=random_harmonic_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
shimmeryness_slider = ttk.Scale(root, from_=0, to=100, variable=shimmeryness_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
initial_noisiness_slider = ttk.Scale(root, from_=0, to=100, variable=initial_noisiness_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
harmonic_unsteadyness_slider = ttk.Scale(root, from_=0, to=100, variable=harmonic_unsteadyness_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
loudness_variation_slider = ttk.Scale(root, from_=0, to=100, variable=loudness_variation_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
comparative_rate_slider = ttk.Scale(root, from_=0, to=100, variable=comparative_rate_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
decay_rate_slider = ttk.Scale(root, from_=-8, to=0, variable=decay_rate_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
clarification_time_slider = ttk.Scale(root, from_=0, to=1000, variable=clarification_time_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
clarification_duration_slider = ttk.Scale(root, from_=0, to=1000, variable=clarification_duration_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
comeup_duration_slider = ttk.Scale(root, from_=0, to=10000, variable=comeup_duration_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)
comedown_duration_slider = ttk.Scale(root, from_=0, to=10000, variable=comedown_duration_value, orient=tk.HORIZONTAL, length=slider_length, command=sliders_update)

volume_slider_label = tk.Label(root, text='')
pitch_slider_label = tk.Label(root, text='')
duration_slider_label = tk.Label(root, text='')
base_frequency_slider_label = tk.Label(root, text='')
evens_slider_label = tk.Label(root, text='')
odds_slider_label = tk.Label(root, text='')
free_bar_slider_label = tk.Label(root, text='')
clamped_bar_slider_label = tk.Label(root, text='')
rectangular_membrane_slider_label = tk.Label(root, text='')
ideal_timpani_slider_label = tk.Label(root, text='')
realistic_timpani_slider_label = tk.Label(root, text='')
#random_harmonic_slider_label = tk.Label(root, text='')
shimmeryness_slider_label = tk.Label(root, text='')
initial_noisiness_slider_label = tk.Label(root, text='')
harmonic_unsteadyness_slider_label = tk.Label(root, text='')
loudness_variation_slider_label = tk.Label(root, text='')
comparative_rate_slider_label = tk.Label(root, text='')
decay_rate_slider_label = tk.Label(root, text='')
clarification_time_slider_label = tk.Label(root, text='')
clarification_duration_slider_label = tk.Label(root, text='')
comeup_duration_slider_label = tk.Label(root, text='')
comedown_duration_slider_label = tk.Label(root, text='')

# Create button for audio generation and playback
load_button = ttk.Button(root, text="Load Instrument (json file that is generated with soundfont)", command=load_instrument)
generate_button = ttk.Button(root, text="Generate and Play Sample", command=generate_and_play_audio)
generate_graph_button = ttk.Button(root, text="Generate, Play and Graph Sample", command=graph_audio)
generate_sfz = ttk.Button(root, text="Generate Soundfont", command=generate_sfz)

# Pack widgets
row = 0
volume_label.grid(row=row, column=0, sticky='E')
volume_slider.grid(row=row, column=1)
volume_slider_label.grid(row=row, column=2)
row += 1
duration_label.grid(row=row, column=0, sticky='E')
duration_slider.grid(row=row, column=1)
duration_slider_label.grid(row=row, column=2)
row += 1
pitch_label.grid(row=row, column=0, sticky='E')
pitch_slider.grid(row=row, column=1)
pitch_slider_label.grid(row=row, column=2)
row += 1
base_frequency_label.grid(row=row, column=0, sticky='E')
base_frequency_slider.grid(row=row, column=1)
base_frequency_slider_label.grid(row=row, column=2)
row += 1
evens_label.grid(row=row, column=0, sticky='E')
evens_slider.grid(row=row, column=1)
evens_slider_label.grid(row=row, column=2)
row += 1
odds_label.grid(row=row, column=0, sticky='E')
odds_slider.grid(row=row, column=1)
odds_slider_label.grid(row=row, column=2)
row += 1
free_bar_label.grid(row=row, column=0, sticky='E')
free_bar_slider.grid(row=row, column=1)
free_bar_slider_label.grid(row=row, column=2)
row += 1
clamped_bar_label.grid(row=row, column=0, sticky='E')
clamped_bar_slider.grid(row=row, column=1)
clamped_bar_slider_label.grid(row=row, column=2)
row += 1
rectangular_membrane_label.grid(row=row, column=0, sticky='E')
rectangular_membrane_slider.grid(row=row, column=1)
rectangular_membrane_slider_label.grid(row=row, column=2)
row += 1
ideal_timpani_label.grid(row=row, column=0, sticky='E')
ideal_timpani_slider.grid(row=row, column=1)
ideal_timpani_slider_label.grid(row=row, column=2)
row += 1
realistic_timpani_label.grid(row=row, column=0, sticky='E')
realistic_timpani_slider.grid(row=row, column=1)
realistic_timpani_slider_label.grid(row=row, column=2)
row += 1
shimmeryness_label.grid(row=row, column=0, sticky='E')
shimmeryness_slider.grid(row=row, column=1)
shimmeryness_slider_label.grid(row=row, column=2)
row += 1
initial_noisiness_label.grid(row=row, column=0, sticky='E')
initial_noisiness_slider.grid(row=row, column=1)
initial_noisiness_slider_label.grid(row=row, column=2)
row += 1
harmonic_unsteadyness_label.grid(row=row, column=0, sticky='E')
harmonic_unsteadyness_slider.grid(row=row, column=1)
harmonic_unsteadyness_slider_label.grid(row=row, column=2)
row += 1
loudness_variation_label.grid(row=row, column=0, sticky='E')
loudness_variation_slider.grid(row=row, column=1)
loudness_variation_slider_label.grid(row=row, column=2)
row += 1
comparative_rate_label.grid(row=row, column=0, sticky='E')
comparative_rate_slider.grid(row=row, column=1)
comparative_rate_slider_label.grid(row=row, column=2)
row += 1
decay_rate_label.grid(row=row, column=0, sticky='E')
decay_rate_slider.grid(row=row, column=1)
decay_rate_slider_label.grid(row=row, column=2)
row += 1
clarification_time_label.grid(row=row, column=0, sticky='E')
clarification_time_slider.grid(row=row, column=1)
clarification_time_slider_label.grid(row=row, column=2)
row += 1
clarification_duration_label.grid(row=row, column=0, sticky='E')
clarification_duration_slider.grid(row=row, column=1)
clarification_duration_slider_label.grid(row=row, column=2)
row += 1
comeup_duration_label.grid(row=row, column=0, sticky='E')
comeup_duration_slider.grid(row=row, column=1)
comeup_duration_slider_label.grid(row=row, column=2)
row += 1
comedown_duration_label.grid(row=row, column=0, sticky='E')
comedown_duration_slider.grid(row=row, column=1)
comedown_duration_slider_label.grid(row=row, column=2)
row += 1
load_button.grid(row=row, column=0, sticky='W')
row += 1
generate_button.grid(row=row, column=0, sticky='W')
row += 1
generate_graph_button.grid(row=row, column=0, sticky='W')
row += 1
generate_sfz.grid(row=row, column=0, sticky='W')

sliders_update(0)
root.mainloop()
