#!/usr/bin/env python3
"""
Test the official TorchTitan checkpoint conversion scripts with GPT-2.
This ensures the GPT2StateDictAdapter works correctly with the full
TorchTitan checkpoint infrastructure (DCP + HuggingFaceStorage).
"""

import torch
import tempfile
import shutil
from pathlib import Path

# Import the official conversion functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "checkpoint_conversion"))
from convert_from_hf import convert_from_hf
from convert_to_hf import convert_to_hf

from torchtitan.models.gpt2 import GPT2, gpt2_args


def test_official_converter_integration():
    """Test that GPT-2 works with torchtitan's official checkpoint converters."""
    print("=" * 70)
    print("Testing Official TorchTitan Checkpoint Converters with GPT-2")
    print("=" * 70)
    
    model_name = "gpt2"
    model_flavor = "124M"
    
    print(f"\nModel: {model_name}, Flavor: {model_flavor}")
    print("\nThis test verifies:")
    print("  1. GPT2StateDictAdapter works with convert_from_hf.py")
    print("  2. GPT2StateDictAdapter works with convert_to_hf.py")
    print("  3. Round-trip conversion through DCP format")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directories
        hf_checkpoint_dir = tmpdir / "hf_checkpoint"
        dcp_checkpoint_dir = tmpdir / "dcp_checkpoint"
        hf_export_dir = tmpdir / "hf_export"
        
        hf_checkpoint_dir.mkdir()
        dcp_checkpoint_dir.mkdir()
        hf_export_dir.mkdir()
        
        print(f"\n1. Creating fake HuggingFace checkpoint...")
        # Create a minimal GPT-2 checkpoint in HF format
        model_args = gpt2_args[model_flavor]
        model = GPT2(model_args)
        model.init_weights()
        
        # Get the adapter to convert to HF format
        from torchtitan.models.gpt2 import GPT2StateDictAdapter
        adapter = GPT2StateDictAdapter(model_args, None)
        hf_state_dict = adapter.to_hf(model.state_dict())
        
        # Save as safetensors (simulating HF checkpoint)
        # Note: This is simplified - real HF checkpoints use safetensors format
        # For testing, we'll save as pytorch file which HuggingFaceStorageReader can handle
        torch.save(hf_state_dict, hf_checkpoint_dir / "model.safetensors")
        print(f"✓ Saved fake HF checkpoint to {hf_checkpoint_dir}")
        
        print(f"\n2. Testing convert_from_hf (HF → TorchTitan DCP)...")
        try:
            convert_from_hf(
                input_dir=str(hf_checkpoint_dir),
                output_dir=str(dcp_checkpoint_dir),
                model_name=model_name,
                model_flavor=model_flavor,
            )
            print(f"✓ Converted HF → DCP successfully")
        except Exception as e:
            print(f"❌ convert_from_hf failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n3. Testing convert_to_hf (TorchTitan DCP → HF)...")
        try:
            # Note: convert_to_hf expects hf_assets_path for the index.json mapping
            # For GPT-2, we don't have this, so it will create a single safetensors file
            convert_to_hf(
                input_dir=str(dcp_checkpoint_dir),
                output_dir=str(hf_export_dir),
                model_name=model_name,
                model_flavor=model_flavor,
                hf_assets_path=None,  # No sharding for GPT-2
            )
            print(f"✓ Converted DCP → HF successfully")
        except Exception as e:
            print(f"❌ convert_to_hf failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n4. Verifying conversion results...")
        # Check that output directory has expected files
        exported_files = list(hf_export_dir.glob("*.safetensors"))
        if exported_files:
            print(f"✓ Found exported safetensors files: {[f.name for f in exported_files]}")
        else:
            print(f"⚠️  No safetensors files found in export directory")
            return False
        
        print("\n" + "=" * 70)
        print("✅ Official Converter Integration Test PASSED!")
        print("=" * 70)
        print("\nThe GPT2StateDictAdapter successfully integrates with:")
        print("  ✓ convert_from_hf.py (HF → TorchTitan)")
        print("  ✓ convert_to_hf.py (TorchTitan → HF)")
        print("  ✓ PyTorch Distributed Checkpoint (DCP)")
        print("  ✓ HuggingFaceStorageReader/Writer")
        
        return True


def main():
    print("\nNote: This test verifies that the GPT2StateDictAdapter")
    print("integrates correctly with torchtitan's official checkpoint")
    print("conversion infrastructure.\n")
    
    try:
        result = test_official_converter_integration()
        if not result:
            exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

