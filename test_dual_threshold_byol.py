#!/usr/bin/env python3
"""
Test script to verify dual_threshold_BYOL implementation
"""
import torch
from omegaconf import OmegaConf
from solo.methods import METHODS

# Create a minimal config
config = {
    "method": "dual_threshold_byol",
    "backbone": {
        "name": "resnet18",
        "kwargs": {}
    },
    "data": {
        "dataset": "cifar10",
        "num_classes": 10,
        "num_large_crops": 2,
        "num_small_crops": 0
    },
    "method_kwargs": {
        "proj_hidden_dim": 4096,
        "proj_output_dim": 256,
        "pred_hidden_dim": 4096,
        "gamma": 1.0,
        "epsilon": 1e-4,
        "e0": 0.1,
        "e1": 0.9,
        "base_tau_momentum": 0.99,
        "final_tau_momentum": 1.0,
        "momentum_classifier": False
    },
    "optimizer": {
        "name": "sgd",
        "batch_size": 256,
        "lr": 0.1,
        "classifier_lr": 0.1,
        "weight_decay": 1e-5,
        "exclude_bias_n_norm_wd": False,
        "kwargs": {}
    },
    "scheduler": {
        "name": "warmup_cosine",
        "min_lr": 0.0,
        "warmup_start_lr": 3e-5,
        "warmup_epochs": 10,
        "lr_decay_steps": None,
        "interval": "step"
    },
    "knn_eval": {
        "enabled": False,
        "k": 20,
        "distance_func": "euclidean"
    },
    "performance": {
        "disable_channel_last": False
    },
    "max_epochs": 100,
    "accumulate_grad_batches": 1,
    "seed": 5
}

cfg = OmegaConf.create(config)

print("=" * 80)
print("Testing dual_threshold_BYOL Implementation")
print("=" * 80)

try:
    # Initialize the model
    print("\n1. Initializing model...")
    model = METHODS["dual_threshold_byol"](cfg)
    print("   ✓ Model initialized successfully")
    
    # Check model components
    print("\n2. Checking model components...")
    assert hasattr(model, 'backbone'), "Missing backbone"
    assert hasattr(model, 'projector'), "Missing projector"
    assert hasattr(model, 'predictor'), "Missing predictor"
    assert hasattr(model, 'momentum_encoder'), "Missing momentum_encoder"
    assert hasattr(model, 'momentum_projector'), "Missing momentum_projector"
    assert hasattr(model, 'iot_loss'), "Missing iot_loss"
    print("   ✓ All required components present")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 8
    # Simulate two augmented views
    X = [torch.randn(batch_size, 3, 32, 32) for _ in range(2)]
    
    model.eval()
    with torch.no_grad():
        out_eval = model(X[0])
        print(f"   ✓ Eval mode forward: output keys = {list(out_eval.keys())}")
    
    model.train()
    out_train = model.forward(X)
    print(f"   ✓ Train mode forward: output keys = {list(out_train.keys())}")
    print(f"   ✓ p_online shape: {out_train['p_online'].shape}")
    print(f"   ✓ z_momentum shape: {out_train['z_momentum'].shape}")
    
    # Test IOT loss
    print("\n4. Testing IOT loss computation...")
    loss = model.iot_loss(out_train['p_online'], out_train['z_momentum'])
    print(f"   ✓ Loss computed: {loss.item():.6f}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    loss.backward()
    print("   ✓ Backward pass successful")
    
    # Check gradients
    print("\n6. Checking gradients...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients computed"
    print("   ✓ Gradients computed successfully")
    
    # Check momentum network parameters are frozen
    print("\n7. Verifying momentum network is frozen...")
    for param in model.momentum_encoder.parameters():
        assert not param.requires_grad, "Momentum encoder should be frozen"
    for param in model.momentum_projector.parameters():
        assert not param.requires_grad, "Momentum projector should be frozen"
    print("   ✓ Momentum networks are correctly frozen")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe dual_threshold_BYOL implementation is working correctly.")
    print("Ready to run full training experiments.")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
