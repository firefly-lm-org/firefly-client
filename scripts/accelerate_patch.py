#!/usr/bin/env python3
"""
accelerate_patch.py - Patch accelerate for bitsandbytes 4-bit compatibility

Required for: bitsandbytes 0.50.0 + transformers 4.44.0 + accelerate 1.14.0

Without this patch: ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
Root cause: from_pretrained() calls dispatch_model() which calls model.to(device) on a quantized model.

Usage:
    python accelerate_patch.py    # Apply patch once
    # Then import transformers normally in remote_run.py
"""
import re

def patch_accelerate():
    path = "/root/miniconda3/lib/python3.12/site-packages/accelerate/big_modeling.py"
    with open(path) as f:
        content = f.read()
    
    if 'is_quantized' in content and 'dispatch_model' in content:
        print("Patch already applied.")
        return True
    
    patched = re.sub(
        r'(    def dispatch_model\(.*?\n)(    )(model\.to\()',
        r'\1    if getattr(model, "is_quantized", False):\n        return model\n\2\3',
        content,
        count=1,
        flags=re.DOTALL
    )
    
    if patched == content:
        print("ERROR: patch pattern did not match!")
        return False
    
    with open(path, "w") as f:
        f.write(patched)
    print("Patch applied successfully to:", path)
    return True

if __name__ == "__main__":
    patch_accelerate()
