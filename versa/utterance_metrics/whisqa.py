

from .WhiSQA.models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall, whisperMetricPredictorEncoderLayersTransformerSmalldim
import torch
import librosa

def whisqa_model_setup(model_type="single", use_gpu: bool = False):
    """
    Setup the WhisQA model for quality assessment.

    Args:
        model_path (str): Path to the pre-trained model.
        use_gpu (bool): Whether to use GPU for inference.

    Returns:
        model: The loaded WhisQA model.
    """

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        model.load_state_dict(
            torch.load("versa_cache/WhiSQA/checkpoints/single_head_model.pt", map_location=device))
    elif model_type == "multi":
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        model.load_state_dict(
            torch.load("WhiSQA/checkpoints/multi_head_model.pt", map_location=device))

    model.eval()
    model = model.to(device)
    return model

def whisqa_metric(model, pred_x,fs):
    """
    Compute the WhisQA metric for quality assessment.

    Args:
        model: The loaded WhisQA model.
        pred_x (np.ndarray): The predicted audio signal.
        fs (int): Sampling frequency of the audio signals.

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    if fs != 16000:
       pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
    pred_x = torch.from_numpy(pred_x).float().unsqueeze(0).to(model.parameters().__next__().device)
    #print(pred_x.shape)
    #input("Press Enter to continue...")  # For debugging purposes, remove in production
    model_prediction = model(pred_x).squeeze(0).cpu().detach()
    #print(model_prediction.shape)
    if len(model_prediction.shape) == 1: #single heads
        model_prediction = model_prediction.item() * 5
        return {"WhiSQA MOS": model_prediction}
    else: #multi heads
        mos = model_prediction[0].item() * 5
        noisiness = model_prediction[1].item() * 5
        coloration = model_prediction[2].item() * 5
        discontinuity = model_prediction[3].item() * 5
        loudness = model_prediction[4].item() * 5

        return {
            "WhiSQA MOS": mos,
            "WhiSQA Noisiness": noisiness,
            "WhiSQA Coloration": coloration,
            "WhiSQA Discontinuity": discontinuity,
            "WhiSQA Loudness": loudness
        }

if __name__ == "__main__":
    import numpy as np
    a = np.random.random(32000)
    model = whisqa_model_setup(use_gpu=False, model_type="single")
    print("metrics: {}".format(whisqa_metric(model, a, 16000)))
