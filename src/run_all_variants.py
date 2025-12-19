#!/usr/bin/env python3
"""
Script to run trainer.py for all three variants (gf, learned, linear) of a given target.

Usage:
    python run_all_variants.py --target funnel --sigma_funnel 3.0
    python run_all_variants.py --target asymmetric_gmm --mean_offset 20.0
    python run_all_variants.py --target funnel --sigma_funnel 3.0 --ntrain 5000

You can override any config parameter that trainer.py supports.
"""

import sys
import subprocess
import argparse
import threading
import time


def main():
    parser = argparse.ArgumentParser(
        description="Run trainer.py for all model variants (gf, learned, linear)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_variants.py --target funnel --sigma_funnel 3.0
  python run_all_variants.py --target asymmetric_gmm --mean_offset 20.0 --ntrain 10000
  python run_all_variants.py --target funnel --sigma_funnel 1.0 --batch_size 2048
        """
    )
    
    # Required argument: target
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        choices=["funnel", "asymmetric_gmm"],
        help="Target distribution to train on"
    )
    
    # Optional config overrides (pass through to trainer.py)
    parser.add_argument("--dim", type=int, help="Override dimension")
    parser.add_argument("--sigma_funnel", type=float, help="Override funnel sigma")
    parser.add_argument("--sigma", type=float, help="Override sigma (initial gaussian)")
    parser.add_argument("--ntrain", type=int, help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--mean_offset", type=float, help="Override mean offset for Asymmetric GMM")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name (will append variant)")
    parser.add_argument("--save_dir", type=str, help="Override save directory")
    parser.add_argument("--run_dir", type=str, help="Override run directory")
    parser.add_argument("--final_eval_samples", type=int, help="Override number of samples for final evaluation")
    parser.add_argument("--final_action_samples", type=int, help="Override number of samples for final action computation")
    parser.add_argument("--divide_by_two", type=str, choices=["True", "False"], help="For Funnel: use exp(y)/2 variance")
    
    args = parser.parse_args()
    
    # Map target to config prefixes
    target_configs = {
        "funnel": ["funnel_gf", "funnel_learned", "funnel_linear"],
        "asymmetric_gmm": ["asymmetric_gmm_learned", "asymmetric_gmm_linear"],
    }
    
    """target_configs = {
        "funnel": ["funnel_learned", "funnel_linear"],
        "asymmetric_gmm": ["asymmetric_gmm_learned", "asymmetric_gmm_linear"],
    }"""
    
    configs = target_configs[args.target]
    
    print(f"\n{'='*70}")
    print(f"Running all variants for target: {args.target}")
    print(f"Configs: {configs}")
    print(f"{'='*70}\n")
    
    # Build command arguments that will be passed to trainer.py
    trainer_args = []
    
    # Add all provided arguments to pass through
    if args.dim is not None:
        trainer_args.extend(["--dim", str(args.dim)])
    if args.sigma_funnel is not None:
        trainer_args.extend(["--sigma_funnel", str(args.sigma_funnel)])
    if args.sigma is not None:
        trainer_args.extend(["--sigma", str(args.sigma)])
    if args.ntrain is not None:
        trainer_args.extend(["--ntrain", str(args.ntrain)])
    if args.batch_size is not None:
        trainer_args.extend(["--batch_size", str(args.batch_size)])
    if args.lr is not None:
        trainer_args.extend(["--lr", str(args.lr)])
    if args.device is not None:
        trainer_args.extend(["--device", args.device])
    if args.mean_offset is not None:
        trainer_args.extend(["--mean_offset", str(args.mean_offset)])
    if args.use_wandb:
        trainer_args.append("--use_wandb")
    if args.wandb_project is not None:
        trainer_args.extend(["--wandb_project", args.wandb_project])
    if args.save_dir is not None:
        trainer_args.extend(["--save_dir", args.save_dir])
    if args.run_dir is not None:
        trainer_args.extend(["--run_dir", args.run_dir])
    if args.final_eval_samples is not None:
        trainer_args.extend(["--final_eval_samples", str(args.final_eval_samples)])
    if args.final_action_samples is not None:
        trainer_args.extend(["--final_action_samples", str(args.final_action_samples)])
    if args.divide_by_two is not None:
        trainer_args.extend(["--divide_by_two", args.divide_by_two])
    
    # Run each config
    failed_configs = []
    processes = {}
    results = {}
    
    def run_config(config, trainer_args, target_args):
        """Run a single config in a thread."""
        print(f"\n{'='*70}")
        print(f"Running: {config}")
        print(f"{'='*70}")
        
        # Build command to run trainer as module
        cmd = ["python", "-m", "src.trainer", "--config", config] + trainer_args
        
        # If wandb_run_name provided, append variant suffix
        if target_args.wandb_run_name:
            wandb_name = f"{target_args.wandb_run_name}_{config.split('_')[-1]}"
            cmd.extend(["--wandb_run_name", wandb_name])
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run the command
        result = subprocess.run(cmd)
        
        results[config] = result.returncode
        
        if result.returncode != 0:
            print(f"\n {config} failed with return code {result.returncode}\n")
        else:
            print(f"\n {config} completed successfully\n")
    
    # Start all configs in parallel threads
    for config in configs:
        thread = threading.Thread(target=run_config, args=(config, trainer_args, args))
        processes[config] = thread
        thread.start()
    
    # Wait for all threads to complete
    for config, thread in processes.items():
        thread.join()
    
    # Check results
    for config, return_code in results.items():
        if return_code != 0:
            failed_configs.append(config)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary for target: {args.target}")
    print(f"{'='*70}")
    print(f"Total configs: {len(configs)}")
    print(f"Successful: {len(configs) - len(failed_configs)}")
    if failed_configs:
        print(f"Failed: {len(failed_configs)}")
        print(f"Failed configs: {', '.join(failed_configs)}")
        return 1
    else:
        print("All configs completed successfully! ")
        return 0


if __name__ == "__main__":
    sys.exit(main())
