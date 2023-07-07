import os
import sys
from argparse import ArgumentParser

import torch
import random
import soundfile
from typing import Optional, Sequence

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../VoiceCloning"))
from InferenceInterfaces.FastSpeech2Interface import InferenceFastSpeech2
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding


class SpeechSynthesizer(object):
    def __init__(
            self,
            language: str,
            model_name: str = "Meta",
            device: Optional[str] = None,
    ) -> None:
        """
        Inputs:
        language: vi
        model_name: So far you can choose from `Meta` (multilingual) and `Vietnamese`
        device: Optional computing device. Default checks if cuda is available.
        """

        self.language = language
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = InferenceFastSpeech2(device=self.device, model_name=model_name)
        self.tts.set_language(language)
        self.utterance_embeddings = dict()
        self.model_name = model_name
        self.vetted_utterances = list()

    def _get_random_vetted_utterance_embedding(self) -> torch.tensor:
        if len(self.vetted_utterances) == 0:
            checkpoints = "/home/..."
            with open(checkpoints + "/prediction_errors.txt", mode="r", encoding="utf8") as f:
                for line in f:
                    audio_path, metric, _ = line.split("\t")
                    metric = float(metric)
                    if metric < 0.1:
                        self.vetted_utterances.append(audio_path)
        utterance = self.vetted_utterances[random.randint(0, len(self.vetted_utterances) - 1)]
        return self._get_utterance_embedding(audio_file=utterance)

    def _get_utterance_embedding(self, audio_file: str) -> torch.tensor:
        if audio_file in self.utterance_embeddings:
            return self.utterance_embeddings[audio_file]
        assert os.path.exists(audio_file)
        device_to_use = "cpu"
        # only cpu supported atm! device if device is not None else self.device
        wave, sr = soundfile.read(audio_file)
        audio_preprocessor = AudioPreprocessor(
            input_sr=sr,
            output_sr=16000,
            cut_silence=True,
            device=device_to_use)
        spec = audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        style_embedding_function = StyleEmbedding()
        utterance_embedding = style_embedding_function(
            spec.unsqueeze(0).to(device_to_use),
            spec_len.unsqueeze(0).to(
                device_to_use)).squeeze()
        self.utterance_embeddings[audio_file] = utterance_embedding
        return utterance_embedding

    def _convert_phones(self, phones: str) -> str:
        phones = phones.replace('A1', "˧")
        phones = phones.replace('A2', "˨˩")
        phones = phones.replace('B1', "˧˥")
        phones = phones.replace('B2', "˦˧˥")
        phones = phones.replace('C1', "˧˩˧")
        phones = phones.replace('C2', "˧˩ʔ˨")
        phones = phones.replace('A1', "˧")
        tf = ArticulatoryCombinedTextFrontend(language=self.language, input_is_phones=self.model_name != "Meta")
        phones = tf.postprocess_phoneme_string(
            phones,
            include_eos_symbol=True,
            for_feature_extraction=True,
            for_plot_labels=False)
        return phones

    def synthesize_and_save(
            self,
            text: Sequence[str],
            output_path: str,
            style_reference_path: Optional[str] = None,
            input_is_phones: bool = False,
            use_random_style_ref: bool = False,
            vary_speaker_style_manually: bool = False,
            device: Optional[str] = None) -> None:
        assert not (style_reference_path is not None and use_random_style_ref is True)
        if style_reference_path is not None:
            embedding = self._get_utterance_embedding(style_reference_path)
            self.tts.set_utterance_embedding(embedding=embedding)
        if use_random_style_ref:
            embedding = self._get_random_vetted_utterance_embedding()
            self.tts.set_utterance_embedding(embedding=embedding)
        duration_sf = 1
        pitch_variance_sf = 1
        energy_variance_sf = 1
        if self.tts.default_utterance_embedding is None or vary_speaker_style_manually:
            duration_sf = random.uniform(0.80, 1.05)
            pitch_variance_sf = random.uniform(0.7, 1.3)
            energy_variance_sf = random.uniform(0.7, 1.3)
        if isinstance(text, str):
            text = [text]
        if input_is_phones:
            text = [self._convert_phones(t) for t in text]
        if "/" in output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.tts.read_to_file(
            text_list=text,
            file_location=output_path,
            pitch_variance_scale=pitch_variance_sf,
            duration_scaling_factor=duration_sf,
            energy_variance_scale=energy_variance_sf,
            input_is_phones=input_is_phones)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--cuda-device', type=str, default="cuda:0")
    parser.add_argument('--file-list', type=str, default="vn-synth-files.txt")
    parser.add_argument('--vary-speaker-style-randomly', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default="Vietnamese")
    args = parser.parse_args()

    tts = SpeechSynthesizer(language="vi", model_name=args.model_name, device=args.cuda_device)
    audio_paths = list()
    phones_list = list()
    reference_styles = list()
    transcripts = list()
    with open(args.file_list, "r") as f:
        for line in f:
            audio_path, transcript, phones, reference_style = line.split("\t")
            reference_style = reference_style.strip("\n").strip("")
            audio_paths.append(audio_path)
            phones_list.append(phones)
            transcripts.append(transcript)
            reference_styles.append(reference_style)
    i = 0
    for audio_path, transcript, phones, reference_style in zip(audio_paths, transcripts, phones_list, reference_styles):
        if not os.path.isfile(audio_path):
            i += 1
            text = transcript if args.model_name == "Meta" else phones
            print(f"audio={audio_path}, text={text}, ref_style={reference_style}")
            tts.synthesize_and_save(
                text,
                output_path=audio_path,
                input_is_phones=not (args.model_name == "Meta"),
                # style_reference_path=reference_style,
                vary_speaker_style_manually=args.vary_speaker_style_randomly,
                use_random_style_ref=True)
        else:
            print(f"File already found: {audio_path}")
        if i > 1000:
            break
    print("Done")
