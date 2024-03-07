# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import json
import shutil

sys.path.append("/src/src/madmom")

from cog import BasePredictor, Input, Path
from typing import Any
import numpy as np
import requests
import subprocess
import shlex
import madmom
from madmom.features import (DBNDownBeatTrackingProcessor, RNNDownBeatProcessor)
import partitura as pt
import pretty_midi as pm

from piano_transcription_inference import PianoTranscription, sample_rate
import librosa

from yt_dlp import YoutubeDL


class Predictor(BasePredictor):
    def download_file_if_url(self, url_str, save_dir="."):
        """
        Download the file at the given URL and save it in the specified directory.
        If the string is not a valid URL or the download fails, the function does nothing.

        :param url_str: The string which may be a URL.
        :param save_dir: Directory where the downloaded file will be saved (defaults to current directory).
        """
        # Check if it looks like a valid URL
        if not (url_str.startswith("http://")
                or url_str.startswith("https://")):
            return url_str

        # Get the file name from the URL or use a default one if not found
        filename = os.path.basename(url_str)
        if not filename:
            filename = "downloaded_file"

        if os.path.exists(os.path.join(save_dir, filename)):
            print(f"using cached {filename} from {url_str}")
            print(
                f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes"
            )
            return f"./{filename}"

        response = requests.get(url_str, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file in the given directory
        with open(os.path.join(save_dir, filename), 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded {filename} from {url_str}")
        print(
            f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes"
        )

        return f"./{filename}"

    def run_separation(self, filename):
        safe_filename = shlex.quote(filename)
        self.run_command(
            f"python -mdemucs --repo ./release_models -n 0d7ed4d7 -o . {safe_filename}"
        )
        return

    # def run_transcription(self, audio_file, midi_file):
    #     safe_audio = shlex.quote(audio_file)
    #     safe_midi = shlex.quote(midi_file)

    #     self.run_command(
    #         f"python -m transkun.transcribe --weight filosax_model --device cuda --segmentHopSize 5 --segmentSize 10 {safe_audio} {safe_midi}"
    #     )

    def get_downbeats_from_syncpoints(self, syncpoints_path):
        syncpoints = json.load(syncpoints_path.open())
        downbeats = [0] + [s[1] for s in syncpoints if len(s) == 2 or s[2] == 0]
        return downbeats

    def preprocess_midi(self, midi_path, downbeats, beats_per_bar = 4):
        mid = pm.PrettyMIDI(str(midi_path))

        tempos = []

        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            t = round(60.0 / ((db2 - db1) / beats_per_bar), 3)
            tempos.append(t)

            tempo_changes = list(zip(downbeats[:-1], tempos))

        # create a new midi file with the correct resolution
        # for loading into Logic Pro X
        #
        # initial tempo is only relevant in cases where the tempo is fixed
        # e.g. played to a click track, in which case you wouldn't need to
        # copy the downbeats over
        out_mid = pm.PrettyMIDI(resolution=960, initial_tempo=60.0)
        for i in range(len(mid.instruments)):
            out_mid.instruments.append(pm.Instrument(program=0))
        out_mid.time_signature_changes.append(pm.TimeSignature(
            beats_per_bar, 4, 0))

        # clear out the existing downbeats
        out_mid._tick_scales = []

        # copy the downbeats from the Filosax midi
        for time, tempo in tempo_changes:
            out_mid._tick_scales.append((int(out_mid.time_to_tick(time)),
                                        60.0 / int(tempo * out_mid.resolution)))
            out_mid._update_tick_to_time(out_mid.get_end_time())

        # with downbeats copied over, we can now add the notes
        for i in range(len(out_mid.instruments)):
            out_mid.instruments[i].notes.extend(mid.instruments[i].notes)

        # add time signature changes at each downbeat
        for time in downbeats:
            out_mid.time_signature_changes.append(pm.TimeSignature(
                beats_per_bar, 4, time))

        # remove overlapping notes
        for instrument in mid.instruments:
            for idx, note in enumerate(instrument.notes):
                for test_note in instrument.notes[idx:]:
                    if test_note.start > note.start and test_note.start < note.end:
                        note.end = test_note.start - 0.001

        # write out the new midi file
        out_mid.write(midi_path)

    def download_audio_as_wav(self, youtube_url):
        # Define the yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': "tmp_audio",  # Set the output filename
        }

        # Use yt-dlp to download the audio
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

    def run_qparse(self, midi_file, beats, beats_per_bar, syncpoints_path):
        if syncpoints_path is None:
            tempo = int(round(60 / np.mean(np.diff(np.array(beats)[:, 0]))))
            print(f"Tempo calculated as {tempo} bpm")

            first_downbeat_idx = 0
            for t, b in beats:
                if b == 1.0:
                    first_downbeat_idx = int(b)
                    break

            downbeats = [
                t for idx, (t, b) in enumerate(beats[first_downbeat_idx:])
                if (idx % beats_per_bar) == 0
            ]

            time_sig = beats_per_bar
            syncpoints_object = [[i, d] for i, d in enumerate(downbeats)]
            syncpoints_json_path = Path(midi_file).with_suffix(".json")
            with open(syncpoints_json_path, 'w') as text_file:
                text_file.write(str(syncpoints_object))
        else:
            syncpoints_json_path = syncpoints_path
            time_sig = beats_per_bar
            downbeats = self.get_downbeats_from_syncpoints(syncpoints_path)
            tempo = int(
                round(60 /
                      np.mean(np.diff(np.array(downbeats) / beats_per_bar))))
            print(f"Tempo calculated as {tempo} bpm")

        beat_output_path = Path(midi_file).with_suffix(".txt")
        np.savetxt(str(beat_output_path.resolve()),
                   downbeats,
                   fmt=['%.3f'],
                   comments='',
                   header=f"""file: {beat_output_path}
tpb: 1
bpb: 1
up: 0
tempo: {tempo}
dur: {max(downbeats)}

""")
        # LD_LIBRARY_PATH="/src:$LD_LIBRARY_PATH" ./monoparse ...
        start = np.round(downbeats[0] - 0.1, 3)
        finish = np.round(downbeats[-1] + 0.1, 3)
        clef = 'G2'  # or F4 for bass

        tmp_mid = pm.PrettyMIDI(midi_file)
        for i in tmp_mid.instruments:
            for n in i.notes:
                if n.start < start:
                    i.notes.remove(n)
                if n.start > finish:
                    i.notes.remove(n)
        
        os.remove(midi_file)
        tmp_mid.write(midi_file)

        mei_output_path = Path(midi_file).with_suffix(".mei")

        cmd = [
            '/src/monoparse', "-v", "5", "-a",
            f"/src/qparse/{time_sig}4-red-charlieparker-omnibook.wta",
            "-barbeat", str(time_sig),
            "-beats",
            str(beat_output_path.resolve()), "-start",
            str(max(0, start)), "-end",
            str(finish), "-config", "/src/qparse/params.ini", "-mono", "-ts",
            f"{int(time_sig)}/4", "-max", "-clef", clef, "-tempo",
            str(tempo), "-m",
            str(Path(midi_file).resolve()), "-o",
            str(mei_output_path.resolve())
        ]

        print(" ".join(cmd))
        qparse_res = self.run_command(cmd, False)

        if qparse_res != 0:
            print("Error running qparse")
            exit()
            
            # add downbeats to midi and preprocess

            # grab the downbeats again to ensure consistent format
            downbeats = self.get_downbeats_from_syncpoints(syncpoints_json_path)

            self.preprocess_midi(midi_file, downbeats, beats_per_bar)

            # use musescore to convert to musicxml directly
            score_output_path = Path(midi_file).with_suffix(".xml")
            cmd = [
                '/usr/bin/musescore3', '-M', '/src/midi_import_settings.xml', '-o', str(score_output_path.resolve()),
                str(midi_file)
            ]
            print(f"running command: {' '.join(cmd)}")
            self.run_command(cmd, False)

            return [syncpoints_json_path, score_output_path]
        else:
            # do some postprocessing to fix the MEI file
            mei_output_path = Path(mei_output_path)

            edited_mei_text = mei_output_path.read_text()
            edited_mei_text = edited_mei_text.replace('num="3" numbase="1"', 'num="3" numbase="2"')

            mei_output_path.write_text(edited_mei_text)

            print("qparse ran successfully")

        score_output_path = Path(midi_file).with_suffix(".xml")
        
        # This method is straightforward but doesn't support nested tuplets
        score = pt.load_score(str(mei_output_path.resolve()))
        pt.save_musicxml(score, str(score_output_path.resolve()))

        # # This method has maximum compatibility but requires a Java runtime
        # # step 1: convert MEI to partwise MusicXML
        # cmd = ['/usr/bin/java', '-jar', 'meicoApp.jar', '-x', 'mei2musicxml.xsl', "timewise.xml", mei_output_path]
        # print(f"running command: {cmd}")
        # self.run_command(cmd, False)

        # # # step 2: convert partwise MusicXML to timewise MusicXML
        # timewise_output_path = Path(midi_file).with_suffix(".timewise.xml")
        # partwise_output_path = Path(midi_file).with_suffix(".partwise.xml")
        # cmd = ['/usr/bin/xsltproc', '-o', partwise_output_path, 'timepart.xsl', timewise_output_path]
        # print(f"running command: {cmd}")
        # self.run_command(cmd, False)

        # # # step 3: add divisions field to the MusicXML file to fix MuseScore import
        # s = partwise_output_path.read_text()
        # s = s.replace("<attributes>", "<attributes>\n            <divisions>960</divisions>")
        # score_output_path.write_text(s)

        return [syncpoints_json_path, score_output_path]

    def run_command(self, command, do_split=True):
        if do_split:
            args = shlex.split(command)
        else:
            args = command

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/src:$LD_LIBRARY_PATH"
        env["QT_QPA_PLATFORM"] = "offscreen"

        result = subprocess.run(args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=env)

        if result.returncode != 0:
            print("Error running command:", result.stdout.decode())
            print("Error running command:", result.stderr.decode())
            print("Command failed - trying to proceed anyway")
        else:
            print("Command output:", result.stdout.decode())

        return result.returncode
    
    def predict(
        self,
        audio_input: Path = Input(description="Piano audio to transcribe",
                                  default=None),
        beats_per_bar: int = Input(
            description="numerator for time signature, default: 4", default=4),
        model_path: str = Input(
            description="Optional URL to specify different model weights",
            default="./model.pth"),
        syncpoints_path: Path = Input(
            description="Optional path to syncpoints file", default=None),
        midi_path: Path = Input(
            description=
            "Option path to midi file - skips audio and runs score layout only",
            default=None),
        skip_separation: bool = Input(description="Skip separation step", default=False),
        file_label: str = Input(description="Optional label for output filename", default=None),
        yt_url: str = Input(description="Optional YouTube URL to fetch audio from - replaces audio_input", default=None),
        device: str = Input(description="Device to run inference on",
                            default="cuda")
    ) -> Any:

        if yt_url is not None:
            # this is where the audio will be saved to
            audio_input = Path("/src/tmp_audio.wav")
            
            # remove old audio file if it exists
            if os.path.exists(audio_input):
                os.remove(audio_input)

            self.download_audio_as_wav(yt_url)

        if midi_path is None:
            midi_intermediate_filename = f"{audio_input.stem}__{file_label}.mid"

            # "python -mdemucs --repo ./release_models -n 0d7ed4d7 -o . Sax-test-mix.wav"
            if not skip_separation:
                separate_stdout = self.run_separation(str(audio_input))
                print(separate_stdout)

                separated_audio_path = str(
                    Path(f"./0d7ed4d7/{audio_input.stem}/Sax.wav"))
            else:
                separated_audio_path = audio_input


            load_audio_start = os.times()[4]
            audio, _ = librosa.core.load(str(separated_audio_path), sr=sample_rate)
            load_audio_end = os.times()[4]

            print(f"Loaded audio in {load_audio_end - load_audio_start} seconds")

            # Transcribe audio
            self.transcriptor = PianoTranscription("Regress_onset_offset_frame_velocity_CRNN",
                                        device='cuda',
                                        checkpoint_path="/src/filosax_25k.pth",
                                        segment_samples=10 * sample_rate,
                                        batch_size=8)
            
            transcribe_start_time = os.times()[4]
            self.transcriptor.transcribe(audio, midi_intermediate_filename)
            transcribe_end_time = os.times()[4]

            print(f"Transcribed audio in {transcribe_end_time - transcribe_start_time} seconds")
        else:
            midi_intermediate_filename = midi_path.stem + f"__{file_label}.mid"
            shutil.copy(midi_path, midi_intermediate_filename)

        if syncpoints_path is None:
            # predict beats

            # get tempo estimate from midi
            midi = pm.PrettyMIDI(midi_intermediate_filename)
            tempo = midi.estimate_tempo()

            print(f"Tempo estimated from MIDI as {tempo} bpm")
            
            # get beats from madmom DBNDownBeatTracker
            processor = DBNDownBeatTrackingProcessor([3, 4], fps=100, min_bpm=int(tempo-15), max_bpm=350, transition_lambda=50)
            in_processor = RNNDownBeatProcessor()

            activations = in_processor(str(audio_input))
            
            beats = processor(activations)
            beats_per_bar = max([int(t) for d,t in beats])
            
            db_times = [d for d,t in beats if int(t) == 1]

            syncpoints_path = Path(midi_intermediate_filename).with_suffix(".json")

            Path(syncpoints_path).write_text(json.dumps([[idx, t] for idx, t in enumerate(db_times)]))

            # estimator = BeatNet(1,
            #                     mode='offline',
            #                     inference_model='DBN',
            #                     plot=[],
            #                     thread=False,
            #                     device='cuda')
            # beats = estimator.process(str(audio_input))
            # # beats is [[time, beat_idx], ...]
            # estimator = None
        else:
            # get the correct naming for the syncpoints file
            syncpoints_out_path = Path(midi_intermediate_filename).with_suffix(".json")
            shutil.copy(syncpoints_path, syncpoints_out_path)
            syncpoints_path = syncpoints_out_path

            beats = []

        syncpoints_json_path, score_path = self.run_qparse(
            midi_intermediate_filename, beats, beats_per_bar, syncpoints_path)

        # to return a list see https://github.com/replicate/cog/blob/main/docs/python.md#returning-a-list
        return [
            Path(midi_intermediate_filename), score_path, syncpoints_json_path
        ]
