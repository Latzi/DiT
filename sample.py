import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # Important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Specify the number of images to generate
    num_images = 10
    save_dir = "generated_images"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):
        # Create sampling noise for each image
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        class_labels = [torch.randint(0, args.num_classes, (1,), device=device)]  # Random class label

        # Setup classifier-free guidance
        z = torch.cat([z, z], 0)  # Duplicate noise for guidance
        y_null = torch.tensor([1000], device=device)  # Assuming the null class is represented by 1000
        y = torch.cat(class_labels + [y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = samples[:1]  # Take the sample corresponding to the real class label
        samples = vae.decode(samples / 0.18215).sample  # Use .sample as an attribute

        # Save the generated image
        save_path = os.path.join(save_dir, f"sample_{i}.png")  # Modify the save path as needed
        save_image(samples, save_path, normalize=True, value_range=(-1, 1))
        print(f"Saved image {i+1} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
