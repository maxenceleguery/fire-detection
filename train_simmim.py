
from simmim.vit import ViT
from simmim.simmim import SimMIM
from dataset import get_unsupervised_train
import torch
from tqdm import tqdm

def main():
    trainloader = get_unsupervised_train(batch_size=64, resize=256)

    vit = ViT(
        image_size=256,
        patch_size=16,
        num_classes=2,
        dim=256,
        depth=18,
        heads=12,
        mlp_dim=512,
    )
    sim = SimMIM(encoder=vit).cuda()
    optimizer = torch.optim.Adam(sim.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(20):
        final_loss = 0
        pbar = tqdm(trainloader, desc="Train SimMIM")
        for batch_idx, (inputs, _, _) in enumerate(pbar):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            loss = sim(inputs)
            loss.backward()
            optimizer.step()
            final_loss += loss
            pbar.set_postfix_str(f"loss={100*final_loss.item()/(batch_idx+1):.4f}")
        scheduler.step()

    torch.save(sim.encoder.state_dict(), "training/simmim-vit_v2.pt")


if __name__ == "__main__":
    main()