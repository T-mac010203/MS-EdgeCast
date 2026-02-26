import os
import torch
from mscrn import Ms_crn as Predictor
from edgeguided import Edge_guided_diffusion
import config as cfg


@torch.no_grad()
def run_predictor(predictor, hist_frames, future_cond):
    """
    Stage-1: MS-CRN
    Args:
        hist_frames: (B, T_in, 1, H, W)
        future_cond: (B, T_cond, 1, H, W)
    Returns:
        coarse_pred: (B, T_out, H, W)
    """
    coarse_pred, _ = predictor(hist_frames, future_cond)
    return coarse_pred


@torch.no_grad()
def run_diffusion(diffusion, hist_frames, coarse_pred, num_blocks=4, block_size=3):
    """
    Stage-2: Edge-guided diffusion
    """
    refined_preds = []

    for block_idx in range(num_blocks):
        # concatenate historical observations and coarse predictions
        inputs = torch.cat(
            [
                hist_frames[:, :, 0],  # (B, T_in, H, W)
                coarse_pred[:, 6 + block_idx * block_size:
                               6 + (block_idx + 1) * block_size]
            ],
            dim=1
        )

        c_emb = torch.full(
            (inputs.size(0),),
            fill_value=block_idx,
            dtype=torch.long,
            device=inputs.device
        )

        refined = diffusion.sample(inputs, c_emb=c_emb)
        refined_preds.append(refined)

    return torch.cat(refined_preds, dim=1)


def load_predictor(device, height):
    model = Predictor(size=32, h=height).to(device)

    ckpt_path = os.path.join(cfg.ckpt_dir, "MSCRN.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    model.eval()
    return model


def load_diffusion(device):
    diffusion = Edge_guided_diffusion(
        T=1000,
        prepath=os.path.join(cfg.ckpt_dir, "BaseDuffusion.pth"),
        path=os.path.join(cfg.ckpt_dir, "EGD.pth"),
    ).to(device)

    diffusion.eval()
    return diffusion


def main():
    device = cfg.device
    print(f"Using device: {device}")

    B = 2
    H = 1024

    # (replace with dataloader)
    hist_frames = torch.rand(B, 6, 1, H, H, device=device)
    future_cond = torch.rand(B, 12, 1, H, H, device=device)

    predictor = load_predictor(device, H)
    diffusion = load_diffusion(device)

    # Stage 1: coarse prediction
    coarse_pred = run_predictor(predictor, hist_frames, future_cond)

    # Stage 2: diffusion refinement
    refined_pred = run_diffusion(diffusion, hist_frames, coarse_pred)


if __name__ == "__main__":
    main()
