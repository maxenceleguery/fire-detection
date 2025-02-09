
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
    sample, _, index = next(iter(trainloader))
    print(vit.cuda()(sample.cuda()).shape)

    sim = SimMIM(encoder=vit).cuda()
    optimizer = torch.optim.Adam(sim.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(10):
        final_loss = 0
        pbar = tqdm(trainloader, desc="Train SimMIM")
        for batch_idx, (inputs, _, _) in enumerate(pbar):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            loss = sim(inputs)
            loss.backward()
            optimizer.step()
            final_loss += loss
        pbar.set_postfix_str(f"loss={final_loss.item():.4f}")

    torch.save(sim.encoder.state_dict(), "training/simmim-vit.pt")


if __name__ == "__main__":
    main()