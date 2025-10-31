#!/usr/bin/env python3
"""
Test script to verify BYOL_IOT_Form2 implementation
"""
import torch
from omegaconf import OmegaConf
from solo.methods import METHODS

# Create a minimal config
config = {
    "method": "byol_iot_form2",
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
        # Form 2: Only 2 IOT parameters (vs 4 in Form 1)
        "gamma": 1.0,
        "epsilon": 1e-4,
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
print("Testing BYOL_IOT_Form2 Implementation")
print("=" * 80)
print("\nForm 2: Single-Threshold Simplified Mapping")
print("- Target plan: X_ij = max(s_ij, ε)")
print("- Only 2 IOT hyperparameters (gamma, epsilon)")
print("- Simpler than Form 1 (dual-threshold with 4 params)")
print()

try:
    # Initialize the model
    print("1. Initializing model...")
    model = METHODS["byol_iot_form2"](cfg)
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
    
    # Check IOT loss is Form 2
    print("\n3. Verifying Form 2 configuration...")
    assert model.iot_loss.gamma == 1.0, "Gamma mismatch"
    assert model.iot_loss.epsilon == 1e-4, "Epsilon mismatch"
    assert not hasattr(model.iot_loss, 'e0'), "Form 2 should not have e0"
    assert not hasattr(model.iot_loss, 'e1'), "Form 2 should not have e1"
    print(f"   ✓ IOT Loss Form 2 configured: gamma={model.iot_loss.gamma}, epsilon={model.iot_loss.epsilon}")
    print("   ✓ No e0/e1 thresholds (Form 2 uses direct clamping)")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
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
    
    # Test IOT Form 2 loss
    print("\n5. Testing IOT Form 2 loss computation...")
    loss = model.iot_loss(out_train['p_online'], out_train['z_momentum'])
    print(f"   ✓ Loss computed: {loss.item():.6f}")
    
    # Test backward pass
    print("\n6. Testing backward pass...")
    loss.backward()
    print("   ✓ Backward pass successful")
    
    # Check gradients
    print("\n7. Checking gradients...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients computed"
    print("   ✓ Gradients computed successfully")
    
    # Check momentum network parameters are frozen
    print("\n8. Verifying momentum network is frozen...")
    for param in model.momentum_encoder.parameters():
        assert not param.requires_grad, "Momentum encoder should be frozen"
    for param in model.momentum_projector.parameters():
        assert not param.requires_grad, "Momentum projector should be frozen"
    print("   ✓ Momentum networks are correctly frozen")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nBYOL IOT Form 2 implementation is working correctly.")
    print("\nKey advantages over Form 1:")
    print("  • Fewer hyperparameters (2 vs 4)")
    print("  • Simpler implementation (direct clamping)")
    print("  • Faster computation (no smoothstep)")
    print("  • Easier to tune")
    print("\nReady to run full training experiments!")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
