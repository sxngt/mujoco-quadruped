# Ubuntu 환경에서 MuJoCo 렌더링 설정 가이드

## 문제 상황
Ubuntu 4080 환경에서 `uv run python train_integrated.py --render` 실행 시 렌더링 창이 나타나지 않음

## 원인 분석
1. **DISPLAY 환경변수 미설정** (가장 흔한 원인)
2. **X11 서버 없음** (헤드리스 서버)
3. **OpenGL 드라이버 문제**
4. **권한 문제**

## 해결 방법

### 1. GUI 데스크톱 환경에서 실행
```bash
# 데스크톱 환경에서 터미널 열고 실행
export DISPLAY=:0
uv run python train_integrated.py --render
```

### 2. SSH X11 포워딩 (로컬에서 원격 접속)
```bash
# 로컬 머신에서 SSH 접속
ssh -X username@ubuntu-server
# 또는 더 안전한 방법
ssh -Y username@ubuntu-server

# 서버에서 실행
uv run python train_integrated.py --render
```

### 3. Xvfb 가상 디스플레이 (헤드리스 서버)
```bash
# Xvfb 설치
sudo apt update
sudo apt install xvfb

# 가상 디스플레이로 실행
xvfb-run -a -s "-screen 0 1024x768x24" uv run python train_integrated.py --render
```

### 4. VNC 원격 데스크톱
```bash
# VNC 서버 설치
sudo apt install tightvncserver

# VNC 서버 시작
vncserver :1 -geometry 1920x1080 -depth 24

# VNC 클라이언트로 접속 후 실행
export DISPLAY=:1
uv run python train_integrated.py --render
```

### 5. Docker with X11 (고급)
```bash
# X11 소켓 공유로 Docker에서 실행
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  your-image uv run python train_integrated.py --render
```

## 진단 명령어

### 환경 확인
```bash
echo "DISPLAY: $DISPLAY"
echo "XDG_SESSION_TYPE: $XDG_SESSION_TYPE"
ps aux | grep -E "(Xorg|wayland)"
```

### OpenGL 확인
```bash
# OpenGL 정보 확인
glxinfo | grep "OpenGL"
# 또는
nvidia-smi  # NVIDIA GPU 확인
```

### 권한 확인
```bash
# X11 접근 권한 확인
xauth list
xhost +local:
```

## 권장 사항

### Ubuntu 4080 서버용 설정
1. **개발 단계**: SSH X11 포워딩 사용
2. **학습 단계**: Xvfb 가상 디스플레이 사용 (GPU 가속 유지)
3. **디버깅**: VNC 원격 데스크톱 사용

### 스크립트 자동화
```bash
#!/bin/bash
# run_with_display.sh
if [ -z "$DISPLAY" ]; then
    echo "DISPLAY 없음, Xvfb 사용"
    xvfb-run -a -s "-screen 0 1920x1080x24" uv run python train_integrated.py --render
else
    echo "DISPLAY 있음, 직접 실행"
    uv run python train_integrated.py --render
fi
```

## 코드에서 자동 감지
현재 코드가 자동으로:
- DISPLAY 환경변수 확인
- 없으면 :0으로 자동 설정
- 적절한 가이드 메시지 출력