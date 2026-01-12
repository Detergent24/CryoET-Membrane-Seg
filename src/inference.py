"""inference  """

import argparse
import os
import numpy as np
import torch
import mrcfile

import config
from model import UNet


def normalize(tomo):
    med = np.median(tomo)
    p5, p95 = np.percentile(tomo, [5, 95])
    tomo = (tomo - med) / (p95 - p5 + 1e-6)
    return np.clip(tomo, -5.0, 5.0).astype(np.float32)


@torch.no_grad()
def infer_tiled(model, tomo):
    Z, Y, X = tomo.shape
    
    BATCH_TILES = config.BATCH_SIZE
    half = config.K_SLICES // 2

    coords = [(y, x) for y in range(0, Y - config.PATCH_SIZE + 1, config.PATCH_SIZE) for x in range(0, X - config.PATCH_SIZE + 1, config.PATCH_SIZE)]

    prob_pad = np.zeros((Z, Y, X), dtype=np.float32)
    accum = np.zeros((Y, X), dtype=np.float32)
    count = np.zeros((Y, X), dtype=np.float32)

    for z in range(half, Z - half):
        accum.fill(0)
        count.fill(0)

        stack = tomo[z - half : z + half + 1]

        tiles, positions = [], []

        for (y0, x0) in coords:
            tiles.append(stack[:, y0:y0 + config.PATCH_SIZE, x0:x0 + config.PATCH_SIZE])
            positions.append((y0, x0))

            if len(tiles) == BATCH_TILES:
                _run_batch(model, tiles, positions, accum, count)
                tiles, positions = [], []

        if tiles:
            _run_batch(model, tiles, positions, accum, count)

        prob_pad[z] = accum / (count + 1e-6)

    return prob_pad[:, :Y, :X]


def _run_batch(model, tiles, positions, accum, count):
    X = torch.from_numpy(np.stack(tiles)).to(config.DEVICE, dtype=torch.float32)
    probs = torch.sigmoid(model(X))[:, 0].detach().cpu().numpy()

    for i, (y0, x0) in enumerate(positions):
        accum[y0:y0 + config.PATCH_SIZE, x0:x0 + config.PATCH_SIZE] += probs[i]
        count[y0:y0 + config.PATCH_SIZE, x0:x0 + config.PATCH_SIZE] += 1
        
        

def save_overlay_pngs(
    tomo: np.ndarray,
    mask: np.ndarray,
):

    import matplotlib.pyplot as plt

    Z = tomo.shape[0]
    half = config.K_SLICES // 2

    z_min = half
    z_max = Z - half - 1

    zs = np.linspace(z_min, z_max, 5, dtype=int)
    zs = np.unique(zs)

    for z in zs:
        raw = tomo[z]

        plt.figure(figsize=(6, 6))
        plt.imshow(raw, cmap="gray")
        plt.imshow(mask[z].astype(np.float32), alpha=0.35)
        plt.axis("off")

        plt.savefig(f"{config.SAVE}/z{z:04d}_overlay.png", dpi=200)
        plt.close()


def see_membrane_3d(mask: np.ndarray):
    import os
    import numpy as np
    from skimage import measure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = (mask > 0).astype(np.uint8)

    m_ds = m[::4, ::4, ::4]


    idx = np.argwhere(m_ds > 0)

    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1
    m_crop = m_ds[z0:z1, y0:y1, x0:x1]

    verts, faces, normals, values = measure.marching_cubes(m_crop.astype(np.float32), level=0.5)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, m_crop.shape[2])
    ax.set_ylim(0, m_crop.shape[1])
    ax.set_zlim(0, m_crop.shape[0])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Membrane surface")
    
    plt.savefig(f"{config.SAVE}/3dvis.png", dpi=200)
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tomo",
        type=str,
        required=True,
        help="Path to input tomogram (.mrc)",
    )
    args = parser.parse_args()

    with mrcfile.open(args.tomo, permissive=True) as m:
        tomo = m.data.astype(np.float32)
        
    tomo = normalize(tomo)
    
    model = UNet()
    
    os.makedirs(config.SAVE, exist_ok=True)

    state = torch.load(f"{config.SAVE}/best_weights.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.to(config.DEVICE)
    model.eval()

    prob = infer_tiled(model, tomo)
    mask = (prob >= 0.5).astype(np.uint8)

    with mrcfile.new(f"{config.SAVE}/prob.mrc", overwrite=True) as m:
        m.set_data(prob.astype(np.float32))
        
    save_overlay_pngs(tomo, mask)
    see_membrane_3d(mask)

    print("done")


if __name__ == "__main__":
    main()
