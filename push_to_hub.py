from models import load_model, MODELS
import torch
from argparse import ArgumentParser
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--hg_repo", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None, choices=MODELS, help="Model")
    parser.add_argument("--DE_size", type=int, default=None, help="Size of the ensemble when a DE model is selected")
    args = parser.parse_args()

    model = load_model(args.model, DE_size=args.DE_size).cpu()
    state_dict = torch.load(args.weights_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)

    model.push_to_hub(args.hg_repo)