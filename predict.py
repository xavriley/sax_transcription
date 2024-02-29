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
from BeatNet.BeatNet import BeatNet
import partitura as pt
import pretty_midi as pm


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

    def run_transcription(self, audio_file, midi_file):
        safe_audio = shlex.quote(audio_file)
        safe_midi = shlex.quote(midi_file)

        self.run_command(
            f"python -m transkun.transcribe --weight filosax_model --device cuda --segmentHopSize 5 --segmentSize 10 {safe_audio} {safe_midi}"
        )

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
        else:
            time_sig = beats_per_bar
            syncpoints_object = json.loads(syncpoints_path.read_text())
            downbeats = [t[1] for t in syncpoints_object[:]]
            tempo = int(
                round(60 /
                      np.mean(np.diff(np.array(downbeats) / beats_per_bar))))
            print(f"Tempo calculated as {tempo} bpm")

        syncpoints_json_path = Path(midi_file).with_suffix(".json")
        with open(syncpoints_json_path, 'w') as text_file:
            text_file.write(str(syncpoints_object))

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
        self.run_command(cmd, False)

        score_output_path = Path(midi_file).with_suffix(".xml")
        
        # This method is straightforward but doesn't support nested tuplets
        score = pt.load_score(str(mei_output_path.resolve()))
        pt.save_musicxml(score, str(score_output_path.resolve()))

        # # This method has maximum compatibility but requires a Java runtime
        # # step 1: convert MEI to partwise MusicXML
        # cmd = ['/usr/bin/java', '-jar', 'meicoApp.jar', '-x', 'mei2musicxml.xsl', "timewise.xml", mei_output_path]
        # print(f"running command: {cmd}")
        # self.run_command(cmd, False)

        # # step 2: convert partwise MusicXML to timewise MusicXML
        # timewise_output_path = Path(midi_file).with_suffix(".timewise.xml")
        # partwise_output_path = Path(midi_file).with_suffix(".partwise.xml")
        # cmd = ['/usr/bin/xsltproc', '-o', partwise_output_path, 'timepart.xsl', timewise_output_path]
        # print(f"running command: {cmd}")
        # self.run_command(cmd, False)

        # # step 3: add divisions field to the MusicXML file to fix MuseScore import
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

        result = subprocess.run(args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=env)

        if result.returncode != 0:
            print("Error running command:", result.stderr.decode())
            print("Command failed - trying to proceed anyway")
        else:
            print("Command output:", result.stdout.decode())

        return

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
        file_label: str = Input(description="Optional label for output filename", default=None),
        device: str = Input(description="Device to run inference on",
                            default="cuda")
    ) -> Any:

        if syncpoints_path is None:
            # predict beats
            estimator = BeatNet(1,
                                mode='offline',
                                inference_model='DBN',
                                plot=[],
                                thread=False,
                                device='cuda')
            beats = estimator.process(str(audio_input))
            # beats is [[time, beat_idx], ...]
            estimator = None
        else:
            beats = []

        if midi_path is None:
            midi_intermediate_filename = f"{audio_input.stem}__{file_label}.mid"

            # "python -mdemucs --repo ./release_models -n 0d7ed4d7 -o . Sax-test-mix.wav"
            separate_stdout = self.run_separation(str(audio_input))
            print(separate_stdout)

            separated_audio_path = str(
                Path(f"./0d7ed4d7/{audio_input.stem}/Sax.wav"))

            # Transcribe audio
            transcribe_start_time = os.times()[4]
            self.run_transcription(separated_audio_path,
                                   midi_intermediate_filename)
            transcribe_end_time = os.times()[4]

            print(
                f"Transcribed audio in {transcribe_end_time - transcribe_start_time} seconds"
            )
        else:
            midi_intermediate_filename = midi_path.stem + f"__{file_label}.mid"
            shutil.copy(midi_path, midi_intermediate_filename)

        syncpoints_json_path, score_path = self.run_qparse(
            midi_intermediate_filename, beats, beats_per_bar, syncpoints_path)

        # to return a list see https://github.com/replicate/cog/blob/main/docs/python.md#returning-a-list
        return [
            Path(midi_intermediate_filename), score_path, syncpoints_json_path
        ]
