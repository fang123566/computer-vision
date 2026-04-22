@echo off
chcp 65001 >nul
echo ============================================================
echo  ASL 手语识别 - 依赖安装脚本
echo ============================================================
echo.

set VENV_PIP=..\.venv\Scripts\pip.exe
set MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
set TRUSTED=pypi.tuna.tsinghua.edu.cn

echo [1/2] 尝试清华镜像安装 (venv)...
"%VENV_PIP%" install flask mediapipe scikit-learn joblib opencv-python pandas ^
    -i %MIRROR% --trusted-host %TRUSTED%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ 安装成功！
    goto done
)

echo.
echo [2/2] 清华源失败，尝试阿里云镜像...
"%VENV_PIP%" install flask mediapipe scikit-learn joblib opencv-python pandas ^
    -i https://mirrors.aliyun.com/pypi/simple/ ^
    --trusted-host mirrors.aliyun.com

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ 安装成功！
    goto done
)

echo.
echo [!] 自动安装失败。请手动执行：
echo     ..\.venv\Scripts\pip install flask mediapipe scikit-learn joblib opencv-python pandas
echo     （在已连接网络或关闭代理后重试）

:done
echo.
echo 安装完成后运行：python run.py
pause
