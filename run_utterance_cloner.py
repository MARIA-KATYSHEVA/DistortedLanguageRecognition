import torch
import whisper

from InferenceInterfaces.UtteranceCloner import UtteranceCloner  

if __name__ == '__main__':
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    uc.clone_utterance(path_to_reference_audio="audios/sample.wav",
                       model = whisper.load_model("small")
                       reference_transcription=model.transcribe(path_to_reference_audio)["text"]',
                       filename_of_result="audios/test_cloned.wav",
                       clone_speaker_identity=True,
                       lang="en")

    uc.biblical_accurate_angel_mode(path_to_reference_audio="audios/sample.wav",
                                    model = whisper.load_model("small"),
                                    reference_transcription=model.transcribe(path_to_reference_audio)["text"],
                                    filename_of_result="audios/test_cloned_angelic.wav",
                                    list_of_speaker_references_for_ensemble=["audios/speaker_references_for_testing/female_high_voice.wav",
                                                                             "audios/speaker_references_for_testing/female_mid_voice.wav",
                                                                             "audios/speaker_references_for_testing/male_low_voice.wav",
                                                                             "audios/LibriTTS/174/168635/174_168635_000019_000001.wav",
                                                                             "audios/test.wav"],
                                    lang="en")
