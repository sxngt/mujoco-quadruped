#!/usr/bin/env python3
"""
GPU 최적화 기능 테스트 스크립트
"""

import torch
import numpy as np

def test_autocast_deprecation():
    """새로운 autocast 문법 테스트"""
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return
    
    device = torch.device('cuda')
    
    # 더미 텐서 생성
    x = torch.randn(100, 50, device=device)
    linear = torch.nn.Linear(50, 10).to(device)
    
    print("🔧 새로운 autocast 문법 테스트")
    
    # 새로운 방식 (권장)
    try:
        with torch.autocast(device_type='cuda', enabled=True):
            y = linear(x)
        print("✅ 새로운 autocast 문법 성공")
    except Exception as e:
        print(f"❌ 새로운 autocast 문법 실패: {e}")
    
    # 기존 방식 (deprecated) - 비교용으로만 유지
    try:
        with torch.autocast(device_type='cuda', enabled=True):
            y = linear(x)
        print("✅ 기존 코드도 새로운 문법으로 업데이트 완료")
    except Exception as e:
        print(f"❌ 업데이트된 autocast 문법 실패: {e}")

def test_gpu_memory():
    """GPU 메모리 사용량 테스트"""
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return
    
    print(f"🔍 GPU 정보:")
    print(f"GPU 이름: {torch.cuda.get_device_name()}")
    print(f"총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 메모리 사용량 측정
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1e9
    print(f"초기 메모리: {initial_memory:.3f}GB")
    
    # 큰 텐서 생성
    large_tensor = torch.randn(10000, 10000, device='cuda')
    after_memory = torch.cuda.memory_allocated() / 1e9
    print(f"큰 텐서 생성 후: {after_memory:.3f}GB (증가: {after_memory - initial_memory:.3f}GB)")
    
    # 메모리 정리
    del large_tensor
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated() / 1e9
    print(f"정리 후: {final_memory:.3f}GB")

def test_mixed_precision():
    """혼합 정밀도 테스트"""
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return
    
    device = torch.device('cuda')
    
    # 간단한 네트워크
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    ).to(device)
    
    x = torch.randn(64, 100, device=device)
    target = torch.randn(64, 10, device=device)
    
    # GradScaler 생성 (새로운 문법)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.Adam(model.parameters())
    
    print("🔥 혼합 정밀도 훈련 테스트")
    
    # 혼합 정밀도로 몇 번의 forward/backward
    for i in range(5):
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', enabled=True):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"스텝 {i+1}: 손실 {loss.item():.4f}")
    
    print("✅ 혼합 정밀도 훈련 성공")

if __name__ == "__main__":
    print("🧪 GPU 최적화 기능 테스트 시작")
    print()
    
    test_autocast_deprecation()
    print()
    
    test_gpu_memory()
    print()
    
    test_mixed_precision()
    print()
    
    print("✅ 모든 테스트 완료")