# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from intraday_tft_nbeats_calibrator import IntradayModel, PlattCalibrator, CONFIG

app = FastAPI()

MODEL_PATH = "best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lazy load
_model = None
_calib = None

class PredictRequest(BaseModel):
    # expects a 2D list: seq_len x features
    seq: list

@app.on_event("startup")
def load_model():
    global _model, _calib
    # ensure CONFIG['seq_len'] matches incoming sequence length
    # create model instance
    # NOTE: set input_dim to the feature dim of your data
    input_dim = len(PredictRequest.__fields__['seq'].type_.__args__[0].__args__[0].__args__) if False else None
    # easier: infer from saved checkpoint or supply explicitly
    # For simplicity below we assume input_dim == 8; change to your actual feature count.
    input_dim = 8
    _model = IntradayModel(input_dim=input_dim, seq_len=CONFIG['seq_len'], cfg=CONFIG)
    _model.to(device)
    _calib = PlattCalibrator().to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        _model.load_state_dict(ckpt['model_state'])
        try:
            _calib.load_state_dict(ckpt['calib_state'])
        except Exception:
            pass
    _model.eval()
    _calib.eval()

@app.post("/predict")
def predict(req: PredictRequest):
    seq = np.array(req.seq, dtype=np.float32)
    # basic shape check
    if seq.ndim != 2:
        return {"error": "seq must be 2D array (seq_len, features)"}
    expected, prob, size = None, None, None
    try:
        expected, prob, size = _model_predict(seq)
    except Exception as e:
        return {"error": str(e)}
    return {"expected": expected.tolist() if isinstance(expected, np.ndarray) else float(expected),
            "prob": prob.tolist() if isinstance(prob, np.ndarray) else float(prob),
            "size": size.tolist() if isinstance(size, np.ndarray) else float(size)}

def _model_predict(seq):
    # reuse predict_and_size from the file or reimplement here
    from intraday_tft_nbeats_calibrator import predict_and_size
    return predict_and_size(_model, _calib, seq, CONFIG, device)
