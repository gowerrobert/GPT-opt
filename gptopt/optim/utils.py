import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from math import sqrt

from transformers import get_cosine_schedule_with_warmup
from .momo import Momo
from .momo_adam import MomoAdam
from .muon import Muon
from .nesgd import NESGD
from .sign_gd import SignGD
from .attn_kq import AttnPDAdamW
from .myadamw import MyAdamW 
# from .sps import SPS
# from .adabound import AdaBoundW
# from .adabelief import AdaBelief
# from .lion import Lion

def get_optimizer(opt_config: dict, lr = 1e-3) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    elif name == 'nadam':
        opt_obj = torch.optim.NAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum_decay': opt_config.get('momentum_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  }
    elif name == 'nadamw':
        opt_obj = torch.optim.NAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum_decay': opt_config.get('momentum_decay', 0),
                  'decoupled_weight_decay': opt_config.get('decoupled_weight_decay', True),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  }
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    
    elif name == 'momo':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': False
                  }
    
    elif name == 'momo-adam':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': False
                  }
        
    elif name == 'momo-star':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': True
                  }
        
    elif name == 'momo-adam-star':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': True
                  }

    elif name == 'muon':
        opt_obj = NESGD
        lmo = True
        prod_norm = "linfty"
        embed_norm = "adam_infty"
        truncate_loss = None

        if "muon_lr_ratio" in opt_config or "muon_lr" in opt_config:
            assert not ("muon_lr_ratio" in opt_config and "muon_lr" in opt_config)
            if "muon_lr_ratio" in opt_config:
                spectral_scale = opt_config["muon_lr_ratio"]
            else:
                spectral_scale = opt_config["muon_lr"] / lr
        else:
            spectral_scale = 1.0

        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'prod_norm': prod_norm,
                  'spectral_scale': spectral_scale,
                  'polar_method': opt_config.get('polar_method', "polar_express"),
                  'embed_norm': embed_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'truncate_loss': truncate_loss,
                  }

    elif name == 'scion':
        opt_obj = NESGD
        lmo = True
        prod_norm = "linfty"
        embed_norm = "linfty"
        truncate_loss = None

        if "muon_lr_ratio" in opt_config or "muon_lr" in opt_config:
            assert not ("muon_lr_ratio" in opt_config and "muon_lr" in opt_config)
            if "muon_lr_ratio" in opt_config:
                spectral_scale = opt_config["muon_lr_ratio"]
            else:
                spectral_scale = opt_config["muon_lr"] / lr
        else:
            spectral_scale = 1.0

        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'prod_norm': prod_norm,
                  'spectral_scale': spectral_scale,
                  'polar_method': opt_config.get('polar_method', "polar_express"),
                  'embed_norm': embed_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'truncate_loss': truncate_loss,
                  }

    elif name == 'muon-momo':
        opt_obj = NESGD
        lmo = True
        prod_norm = "linfty"
        embed_norm = "linfty"
        truncate_loss = opt_config.get('truncate_loss', 3.2)

        if "muon_lr_ratio" in opt_config or "muon_lr" in opt_config:
            assert not ("muon_lr_ratio" in opt_config and "muon_lr" in opt_config)
            if "muon_lr_ratio" in opt_config:
                spectral_scale = opt_config["muon_lr_ratio"]
            else:
                spectral_scale = opt_config["muon_lr"] / lr
        else:
            spectral_scale = 1.0

        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'prod_norm': prod_norm,
                  'spectral_scale': spectral_scale,
                  'polar_method': opt_config.get('polar_method', "polar_express"),
                  'embed_norm': embed_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'truncate_loss': truncate_loss,
                  }

    elif name == 'muonmax-momo':
        opt_obj = NESGD
        lmo = False
        prod_norm = "hybrid"
        embed_norm = "adam_2"
        truncate_loss = opt_config.get('truncate_loss', 3.2)

        if "muon_lr_ratio" in opt_config or "muon_lr" in opt_config:
            assert not ("muon_lr_ratio" in opt_config and "muon_lr" in opt_config)
            if "muon_lr_ratio" in opt_config:
                spectral_scale = sqrt(opt_config["muon_lr_ratio"])
            else:
                spectral_scale = sqrt(opt_config["muon_lr"] / lr)
        else:
            spectral_scale = 1.0

        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'prod_norm': prod_norm,
                  'spectral_scale': spectral_scale,
                  'polar_method': opt_config.get('polar_method', "polar_express"),
                  'embed_norm': embed_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'truncate_loss': truncate_loss,
                  }

    elif name == 'sign-gd':
        opt_obj = SignGD
        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': opt_config.get('nesterov', False),
                  'lmo': True
                  }

    elif name == 'sign-gd-nonlmo':
        opt_obj = SignGD
        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': opt_config.get('nesterov', False),
                  'lmo': False
                  }

    elif name == "attn_pd_adamw":
        opt_obj = AttnPDAdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)), 
                  'rho_over_lr': opt_config.get('rho_over_lr', 10),
                  'attn_max_iter': opt_config.get('attn_max_iter', 100),
                  'pd_type': opt_config.get('pd_type', 'pdhg'),
                  'momentum': opt_config.get('momentum', False),
                  'diag_scaling': opt_config.get('diag_scaling', True), 
                  'warm_start': opt_config.get('warm_start', False),
                  'lsqr_max_iter': opt_config.get('lsqr_max_iter', 100)
                  } 
    elif name == "attn_rehpdhg_adamw":
        opt_obj = AttnPDAdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)), 
                  'rho_over_lr': opt_config.get('rho_over_lr', 10),
                  'attn_max_iter': opt_config.get('attn_max_iter', 100),
                  'pd_type': opt_config.get('pd_type', 'pdhg'),
                  'momentum': opt_config.get('momentum', False),
                  'diag_scaling': opt_config.get('diag_scaling', True), 
                  'reflected_halpern': opt_config.get('reflected_halpern', True), 
                  'warm_start': opt_config.get('warm_start', False),
                  'enable_restart': opt_config.get('enable_restart', False),
                  'lsqr_max_iter': opt_config.get('lsqr_max_iter', 100)
                  }
    elif name == "attn_fista_adamw":
        opt_obj = AttnPDAdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)), 
                  'rho_over_lr': opt_config.get('rho_over_lr', 10),
                  'attn_max_iter': opt_config.get('attn_max_iter', 100),
                  'pd_type': opt_config.get('pd_type', 'fista'),
                  'momentum': opt_config.get('momentum', False),
                  'attn_momentum': opt_config.get('attn_momentum', ""),
                  "mu_frac": opt_config.get("mu_frac", 0.1),
                  "lsqr_max_iter": opt_config.get("lsqr_max_iter", 100)
                  }
    elif name == 'my_adamw':
        opt_obj = MyAdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                    #   'lr_schedule': opt_config.get('lr_schedule', 'constant-linear'),
                    #   'fused': True,
                    #   'warm_up_fraction': opt_config.get('warm_up_fraction', 0.4),
                  }
    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_scheduler(config: dict, opt: torch.optim.Optimizer, total_iterations = None) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('lr_schedule', 'constant')
    
    if name == 'constant':
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    elif 'warm-up-cosine' in name:
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations) 
        scheduler = get_cosine_schedule_with_warmup(
                    opt,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_iterations
                    )


    elif 'constant-linear' in name:  # New scheduler
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations)

        def get_lr(step):
            if step < num_warmup_steps:
                return 1.0  # Constant learning rate during warm-up
            else:
                # Linearly decay after warm-up
                return max(0.1, 1.0 - (step - num_warmup_steps) / (total_iterations - num_warmup_steps))

        scheduler = LambdaLR(opt, lr_lambda=get_lr)
        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler
