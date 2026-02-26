import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import utils
from models.vqvae import VQVAE
from models.vae import VAE


def run(cfg):
    cfg.filename = cfg.filename or utils.readable_timestamp()
    device = torch.device("cpu" if cfg.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and not cfg.cpu and torch.cuda.is_available():
        print("Note: CUDA available but may be incompatible. Use cpu=true if you see cuDNN errors.", flush=True)

    if cfg.save:
        prefix = "vae" if cfg.model == "vae" else "vqvae"
        save_dir = cfg.output_dir if hasattr(cfg, 'output_dir') and cfg.output_dir else './results'
        print(f'Results will be saved in {save_dir}/{prefix}_data_{cfg.filename}.pth', flush=True)

    print('Loading data...', flush=True)
    training_data, validation_data, training_loader, validation_loader, test_loader, x_train_var = utils.load_data_and_data_loaders(
        cfg.dataset, cfg.batch_size, data_root=cfg.data_root)
    print(f'Data loaded: train={len(training_data)}, val={len(validation_data)}', flush=True)

    use_wandb = False
    wandb_run_name = f"{cfg.model}_embedding_dim_{cfg.embedding_dim}_n_embeddings_{cfg.n_embeddings}_{cfg.dataset}"
    if not cfg.debug:
        try:
            import wandb
            use_wandb = True
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run or wandb_run_name, config=config_dict)
        except ImportError:
            pass

    def log_print(msg, force_flush=False):
        print(msg, flush=cfg.debug or force_flush)

    t0 = time.time()
    print(f'Creating {cfg.model} on {device}...', flush=True)
    if cfg.model == "vae":
        model = VAE(cfg.n_hiddens, cfg.n_residual_hiddens,
                    cfg.n_residual_layers, cfg.n_embeddings, cfg.embedding_dim, cfg.beta).to(device)
    else:
        model = VQVAE(cfg.n_hiddens, cfg.n_residual_hiddens,
                      cfg.n_residual_layers, cfg.n_embeddings, cfg.embedding_dim, cfg.beta).to(device)
    print(f'Model created ({time.time()-t0:.1f}s)', flush=True)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, amsgrad=True)
    model.train()

    results = {'n_updates': 0, 'recon_errors': [], 'loss_vals': [], 'perplexities': []}

    def batch_iterator(loader):
        """Yield batches; when exhausted, reshuffle (new epoch) and continue."""
        while True:
            for batch in loader:
                yield batch

    batch_iter = batch_iterator(training_loader)

    log_print(f"Training started: {cfg.n_steps} steps, device={device}", force_flush=True)
    if cfg.debug:
        log_print("DEBUG mode: wandb disabled, output every step")

    for step in range(cfg.n_steps):
        if step == 0:
            t1 = time.time()
        (x, _) = next(batch_iter)
        x = x.to(device)
        if step == 0:
            print(f'First batch fetched ({time.time()-t1:.1f}s)', flush=True)
        optimizer.zero_grad()

        reg_loss, x_hat, aux_metric = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + reg_loss

        loss.backward()
        optimizer.step()
        if step == 0:
            print(f'First step done ({time.time()-t1:.1f}s total)', flush=True)

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(aux_metric.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = step

        log_every = 1 if cfg.debug else cfg.log_interval
        aux_name = "KL" if cfg.model == "vae" else "Perplexity"
        if step % log_every == 0:
            n = min(log_every, len(results["recon_errors"]))
            avg_recon = np.mean(results["recon_errors"][-n:])
            avg_loss = np.mean(results["loss_vals"][-n:])
            avg_aux = np.mean(results["perplexities"][-n:])
            if use_wandb:
                log_d = {"train/recon_loss": avg_recon, "train/loss": avg_loss, "step": step}
                log_d["train/kl_loss" if cfg.model == "vae" else "train/perplexity"] = avg_aux
                wandb.log(log_d)

            if cfg.save and step % cfg.log_interval == 0:
                hp = OmegaConf.to_container(cfg, resolve=True)
                output_dir = getattr(cfg, 'output_dir', None)
                utils.save_model_and_results(model, results, hp, cfg.filename, model_type=cfg.model, output_dir=output_dir)

            log_print(f'Step #{step}  Recon: {avg_recon:.4f}  Loss: {avg_loss:.4f}  {aux_name}: {avg_aux:.4f}')

        if cfg.eval_interval > 0 and (step + 1) % cfg.eval_interval == 0:
            val_recon, val_aux = utils.evaluate(model, validation_loader, x_train_var, device)
            log_dict = {"val/recon_loss": val_recon, "step": step + 1}
            log_dict["val/kl_loss" if cfg.model == "vae" else "val/perplexity"] = val_aux
            if test_loader is not None:
                test_recon, test_aux = utils.evaluate(model, test_loader, x_train_var, device)
                log_dict["test/recon_loss"] = test_recon
                log_dict["test/kl_loss" if cfg.model == "vae" else "test/perplexity"] = test_aux
                log_print(f'  Val  Recon: {val_recon:.4f}  {aux_name}: {val_aux:.4f}')
                log_print(f'  Test Recon: {test_recon:.4f}  {aux_name}: {test_aux:.4f}')
            else:
                log_print(f'  Val  Recon: {val_recon:.4f}  {aux_name}: {val_aux:.4f}')

            if use_wandb:
                x_val, _ = next(iter(validation_loader))
                x_val = x_val.to(device)
                with torch.no_grad():
                    _, x_hat_val, _ = model(x_val)
                n_vis = min(8, x_val.shape[0])
                origin_imgs = (x_val[:n_vis].cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                recon_imgs = (x_hat_val[:n_vis].cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                # hstack origin (left) and recon (right) for each sample
                combined = (np.stack([origin_imgs, recon_imgs], axis=0) * 255).astype(np.uint8)
                log_dict["images/origin_recon"] = [wandb.Image(np.hstack([combined[0, k], combined[1, k]]), caption=f"left: origin, right: recon #{k}") for k in range(n_vis)]
                wandb.log(log_dict)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    @hydra.main(config_path="conf", config_name="config", version_base=None)
    def main(cfg):
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.output_dir = HydraConfig.get().runtime.output_dir
        run(cfg)
    main()
