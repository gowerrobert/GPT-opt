import torch
import time 
import contextlib
import torch.distributed as dist
from gptopt.utils import get_worker_info, save_checkpoint, load_checkpoint
import json
from gptopt.gpt_model import CausalSelfAttention
import numpy as np
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict, deque


typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():
    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []
        self.kq_max = []
        self.val_kq_max = []


def eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt):
    world_size, rank, local_rank, device  = get_worker_info()
    model.eval()
    val_loss, counter = 0., 0
    with torch.inference_mode():
        for batch in val_dataloader:
            with autocast_ctxt:
                output = model(input_ids=batch[0], labels=batch[1])
                val_loss += (output.loss if hasattr(output, "loss") else output[1])
            counter += 1
            if (val_accum_steps != 0) & (counter >= val_accum_steps): break
    # Avoid constructing a new tensor from a tensor; detach and move
    val_loss = (val_loss.detach() / counter).to(device)
    if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"Validation Loss: {val_loss.item()}")
    model.train()
    return val_loss


def train(train_dataloader, val_dataloader, model, optimizer, training_params, logging_params, scheduler=None, ckpt_dir="", wandb_run=None):
    
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    optimizer_name = optimizer.__class__.__name__ 
    if 'momo' in optimizer_name.lower() or 'nesgd' in optimizer_name.lower():
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")

    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')
    log_inner_curves = total_iterations // logging_params.get('inner_iter_curves', 50)
    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=None)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")

    # Training loop
    max_curves_per_layer = logging_params.get("max_curves_per_layer", 50)
    curve_hist = {"r_rel":defaultdict(lambda: deque(maxlen=max_curves_per_layer)), 
                  "z":defaultdict(lambda: deque(maxlen=max_curves_per_layer)), 
                  "y":defaultdict(lambda: deque(maxlen=max_curves_per_layer))} 
    
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)  # micro-step counter
        opt_step = 0 if load_ckpt_step == 0 else int(load_ckpt_step) // grad_accum_steps  # optimizer step counter
        optimizer.zero_grad(set_to_none=True)
        if step != 1 and master_process:
            print(train_dataloader.get_state())
            print(f"Resuming from micro_step={step}, opt_step={opt_step}")
        residuals_n_layers = {}
        for batch in train_dataloader:
            with autocast_ctxt:
                output = model(input_ids=batch[0], labels=batch[1])
                loss = (output.loss if hasattr(output, "loss") else output[1])
                loss /= grad_accum_steps

            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                if "AttnPDAdamW" == optimizer_name:
                    if pass_loss:
                        loss, residuals_n_layers = optimizer.step(closure=None, loss=loss_accum)
                    else:
                        loss, residuals_n_layers = optimizer.step()
                else:
                    if pass_loss:
                        optimizer.step(closure=None, loss=loss_accum)
                    else:
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                # Count an optimizer step
                opt_step += 1
                if master_process and wandb_run is not None:
                    wandb_log_dict = {
                        "train/loss": loss_accum.item(), 
                        "train/grad_norm": norm.item(),
                        "train/step_time": step_time,
                        "train/step": opt_step,
                        "train/micro_step": step
                    } 
                    
                    if ((opt_step-1) % log_inner_curves == 0 or step == len(train_dataloader)-1) and residuals_n_layers:
                        for layer_idx, res in residuals_n_layers.items():  
                            x  = list(range(len(res["r_rel"])))

                            for curve_type, c_hist in curve_hist.items():
                                c_hist[int(layer_idx)].append((opt_step-1, x, res["r_rel"])) 
                                # overlay ALL stored curves for this layer in one plot: (opt_step, x_list, y_list)
                                hist = list(c_hist[int(layer_idx)])
                                xs = [h[1] for h in hist]
                                ys = [h[2] for h in hist]
                                keys = [f"step_{h[0]}" for h in hist]
                                wandb_log_dict[f"train/attn_kq_opt_overlay/layer_{int(layer_idx):02d}/{curve_type}"] = wandb.plot.line_series(
                                    xs=xs, ys=ys, keys=keys, title=f"Attn {curve_type} (layer {int(layer_idx):02d})", xname="attn iteration")
 
  
                    if getattr(model.config, "record_kq_max", False):
                        base_model = getattr(model, "module", model)
                        kq_max = None
                        for m in base_model.modules():
                            if isinstance(m, CausalSelfAttention) and getattr(m, "kq_max", None) is not None:
                                v = m.kq_max
                                if kq_max is None or v > kq_max:
                                    kq_max = v
                        if kq_max is not None:
                            wandb_log_dict["train/kq_max"] = kq_max
                            logger.kq_max.append(kq_max)
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    wandb_run.log(wandb_log_dict, step=opt_step)
                logger.step_times.append(step_time)  # Are these different across ranks?
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (opt_step % logging_params['log_step'] == 0) & master_process:
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {opt_step} of {total_iterations} (optimizer steps).")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f} | Accum: {grad_accum_steps} micro-steps/opt-step")
                    
                if (opt_step % logging_params['val_step'] == 0):
                    val_loss = eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt)
                    if master_process and wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss.item(), "val/step": opt_step, "val/micro_step": step})
                    logger.val_losses.append(val_loss.item())

                if (opt_step % logging_params['save_ckpt_step'] == 0) & (ckpt_dir != ""):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler, logging_params['keep_last'])
                    
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time() 
            step += 1
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set, and save final values
        val_dataloader.reset()
        val_loss = eval_validation_loss(model, val_dataloader, 0, autocast_ctxt)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}") 
        if getattr(model.config, "record_kq_max", False):
            base_model = getattr(model, "module", model)
            kq_max = None
            for m in base_model.modules():
                if isinstance(m, CausalSelfAttention) and getattr(m, "kq_max", None) is not None:
                    v = m.kq_max
                    if kq_max is None or v > kq_max:
                        kq_max = v  
                        logger.val_kq_max.append(kq_max)     
        if (ckpt_dir != ""):
            save_checkpoint(ckpt_dir, step, model, optimizer, logger.losses[-1],
                        train_dataloader, scheduler, logging_params['keep_last'])        
            if master_process:
                with open(ckpt_dir + '/log.json', 'w') as file:
                    json.dump(logger.__dict__, file)
        if master_process and wandb_run is not None:
            wandb_log_dict = {
                "val/loss": val_loss.item(),
                "val/step": opt_step,
                "val/micro_step": step,
                "train/loss": logger.losses[-1],
                "train/step": opt_step,
                "train/micro_step": step,
            }
            if getattr(model.config, "record_kq_max", False) and kq_max is not None:
                    wandb_log_dict["val/kq_max"] = kq_max
            wandb_run.log(wandb_log_dict)

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger
