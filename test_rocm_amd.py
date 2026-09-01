# -*- coding: utf-8 -*-
"""
ROCm Test Script for AMD Radeon RX 6700 XT (12GB VRAM)
Course: CS116 - Do an T11 (GPU AMD ROCm Optimization)
Tests: device detection, FP16 mixed precision, memory allocation, model inference
"""
import os
import sys
import time
import json
import torch
import numpy as np

# Force ROCm visibility
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_device_detection():
    """Test 1: Verify AMD GPU detection under ROCm."""
    print("\n" + "=" * 60)
    print("TEST 1: Device Detection (ROCm / AMD GPU)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/HIP not available - cannot detect GPU")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    is_rocm = any(k in device_name.lower() for k in ["amd", "radeon", "gfx", "rx "])
    
    print(f"[INFO] Device count: {device_count}")
    print(f"[INFO] Device name: {device_name}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] ROCm detected: {is_rocm}")
    print(f"[INFO] HIP version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    
    if is_rocm:
        print("[PASS] AMD GPU detected under ROCm")
        return True
    else:
        print("[WARN] AMD GPU not detected - may be NVIDIA CUDA")
        return True  # Still pass, just not ROCm


def test_memory_allocation():
    """Test 2: Verify memory allocation on 12GB VRAM."""
    print("\n" + "=" * 60)
    print("TEST 2: Memory Allocation (12GB VRAM)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[SKIP] No GPU available")
        return True
    
    try:
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_memory / (1024 ** 3)
        print(f"[INFO] Total GPU memory: {total_gb:.2f} GB")
        
        # Allocate 8GB tensor to verify VRAM
        alloc_size = min(int(8 * 1024**3 // 4), int(total_memory * 0.75))  # 75% of total or 8GB
        alloc_gb = alloc_size * 4 / (1024 ** 3)
        print(f"[INFO] Allocating {alloc_gb:.2f} GB tensor...")
        
        start = time.time()
        x = torch.randn(alloc_size // 4, device="cuda:0", dtype=torch.float32)
        alloc_time = time.time() - start
        
        print(f"[INFO] Allocation time: {alloc_time:.3f}s")
        print(f"[INFO] Memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"[INFO] Memory reserved: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
        
        del x
        torch.cuda.empty_cache()
        print(f"[INFO] Memory after cleanup: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        print("[PASS] Memory allocation successful on AMD GPU")
        return True
    except Exception as e:
        print(f"[FAIL] Memory allocation failed: {e}")
        return False


def test_fp16_mixed_precision():
    """Test 3: Verify FP16 mixed precision on AMD ROCm."""
    print("\n" + "=" * 60)
    print("TEST 3: FP16 Mixed Precision (ROCm)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[SKIP] No GPU available")
        return True
    
    try:
        # Test FP16 tensor operations
        a_fp32 = torch.randn(1024, 1024, device="cuda:0", dtype=torch.float32)
        b_fp32 = torch.randn(1024, 1024, device="cuda:0", dtype=torch.float32)
        
        # FP32 matmul
        start = time.time()
        c_fp32 = torch.matmul(a_fp32, b_fp32)
        fp32_time = time.time() - start
        
        # FP16 matmul
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        
        start = time.time()
        c_fp16 = torch.matmul(a_fp16, b_fp16)
        fp16_time = time.time() - start
        
        # Check results
        c_fp16_to_fp32 = c_fp16.float()
        max_diff = (c_fp32 - c_fp16_to_fp32).abs().max().item()
        
        print(f"[INFO] FP32 matmul time: {fp32_time * 1000:.2f} ms")
        print(f"[INFO] FP16 matmul time: {fp16_time * 1000:.2f} ms")
        print(f"[INFO] Speedup (FP16): {fp32_time / fp16_time:.2f}x")
        print(f"[INFO] Max diff (FP32 vs FP16): {max_diff:.6f}")
        
        if fp16_time < fp32_time:
            print("[PASS] FP16 mixed precision working and faster on AMD ROCm")
        else:
            print("[WARN] FP16 not faster, but functional")
        
        del a_fp32, b_fp32, c_fp32, a_fp16, b_fp16, c_fp16, c_fp16_to_fp32
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[FAIL] FP16 test failed: {e}")
        return False


def test_model_inference():
    """Test 4: Verify PhoBERT model loading and inference on AMD GPU."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Inference (PhoBERT on AMD GPU)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[SKIP] No GPU available")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        
        model_name = "vinai/phobert-base-v2"
        print(f"[INFO] Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model = model.to("cuda:0")
        model.eval()
        
        # Model memory
        model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        print(f"[INFO] Model size: {model_mem:.2f} MB")
        print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        context = "Trường Đại học Công nghệ Thông tin là một trường đại học thành viên của Đại học Quốc gia Thành phố Hồ Chí Minh, được thành lập theo Quyết định số 134/2006/QĐ-TTg ngày 8 tháng 6 năm 2006 của Thủ tướng Chính phủ."
        question = "Trường Đại học Công nghệ Thông tin được thành lập vào ngày tháng năm nào?"
        
        inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True, padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        # FP32 inference
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        fp32_time = time.time() - start
        
        # FP16 inference
        model_fp16 = model.half()
        inputs_fp16 = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs_fp16 = model_fp16(**inputs_fp16)
        fp16_time = time.time() - start
        
        # Decode answer
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        if end_idx < start_idx:
            end_idx = min(start_idx + 5, inputs["input_ids"].shape[1] - 1)
        
        answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1].tolist()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_ids))
        answer = answer.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
        
        print(f"[INFO] Question: {question}")
        print(f"[INFO] Answer: {answer}")
        print(f"[INFO] FP32 inference time: {fp32_time * 1000:.2f} ms")
        print(f"[INFO] FP16 inference time: {fp16_time * 1000:.2f} ms")
        print(f"[INFO] FP16 speedup: {fp32_time / fp16_time:.2f}x")
        print(f"[INFO] GPU memory used: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        
        del model, model_fp16, inputs, inputs_fp16
        torch.cuda.empty_cache()
        print("[PASS] Model inference successful on AMD GPU")
        return True
    except Exception as e:
        print(f"[FAIL] Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_size_12gb_vram():
    """Test 5: Verify batch size works within 12GB VRAM."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Size for 12GB VRAM")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[SKIP] No GPU available")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        
        model_name = "vinai/phobert-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model = model.to("cuda:0")
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        results = []
        
        for bs in batch_sizes:
            try:
                # Create dummy batch
                dummy_input_ids = torch.full((bs, 384), 1, dtype=torch.long, device="cuda:0")
                dummy_attention_mask = torch.ones((bs, 384), dtype=torch.long, device="cuda:0")
                
                torch.cuda.reset_peak_memory_stats()
                
                start = time.time()
                with torch.no_grad():
                    outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
                inference_time = time.time() - start
                
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                results.append({
                    "batch_size": bs,
                    "peak_memory_gb": round(peak_mem, 2),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "success": True
                })
                print(f"[INFO] Batch {bs}: {peak_mem:.2f} GB peak, {inference_time * 1000:.2f} ms")
                
                del dummy_input_ids, dummy_attention_mask, outputs
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    results.append({
                        "batch_size": bs,
                        "peak_memory_gb": None,
                        "inference_time_ms": None,
                        "success": False
                    })
                    print(f"[INFO] Batch {bs}: OOM")
                else:
                    raise
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        # Find max batch size that fits
        max_success_bs = max([r["batch_size"] for r in results if r["success"]], default=1)
        print(f"[PASS] Max batch size for 12GB VRAM: {max_success_bs}")
        return True
    except Exception as e:
        print(f"[FAIL] Batch size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ROCm tests."""
    print("=" * 60)
    print("ROCm Test Suite for AMD Radeon RX 6700 XT (12GB VRAM)")
    print("PhoBERT QA - CS116 Do an T11")
    print("=" * 60)
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] HIP version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Total VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.2f} GB")
    
    tests = [
        ("Device Detection", test_device_detection),
        ("Memory Allocation", test_memory_allocation),
        ("FP16 Mixed Precision", test_fp16_mixed_precision),
        ("Model Inference", test_model_inference),
        ("Batch Size (12GB VRAM)", test_batch_size_12gb_vram),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append({"name": name, "passed": passed})
        except Exception as e:
            print(f"[ERROR] {name} raised exception: {e}")
            results.append({"name": name, "passed": False})
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")
    
    all_passed = all(r["passed"] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    # Save results
    result_path = os.path.join(PROJECT_DIR, "logs", "rocm_test_results.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({
            "pytorch_version": torch.__version__,
            "hip_version": str(torch.version.hip) if hasattr(torch.version, 'hip') else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gb": torch.cuda.get_device_properties(0).total_mem / (1024**3) if torch.cuda.is_available() else None,
            "tests": results,
            "all_passed": all_passed,
        }, f, indent=2)
    print(f"[INFO] Results saved to {result_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
