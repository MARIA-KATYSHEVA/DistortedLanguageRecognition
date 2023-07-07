import torch
import whisper

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

if __name__ == '__main__':
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    # What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
    uc.clone_utterance(path_to_reference_audio_for_intonation="audios/sample1.wav",
                       path_to_reference_audio_for_voice="audios/sample2.wav",  
                       # the two reference audios can be the same, but don't have to be
                       model = whisper.load_model("small"),
                       transcription_of_intonation_reference=model.transcribe(path_to_reference_audio_for_intonation)["text"],
                       filename_of_result="audios/test_cloned.wav",
                       lang="en")

    # Have multiple voices speak with the exact same intonation simultaneously
    uc.biblical_accurate_angel_mode(path_to_reference_audio_for_intonation="audios/sample1.wav",
                                    model = whisper.load_model("small"),
                                    transcription_of_intonation_reference=model.transcribe(path_to_reference_audio_for_intonation)["text"],
                                    list_of_speaker_references_for_ensemble="audios/test_cloned.wav",
                                                                             #"audios/speaker_references_for_testing/female_mid_voice.wav",
                                                                             #"audios/speaker_references_for_testing/male_low_voice.wav"],
                                    filename_of_result="audios/test_cloned_angelic.wav",
                                    lang="en")
