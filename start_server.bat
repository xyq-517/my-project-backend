@echo off
title UNet分割服务 - 端口5002
echo ============================================================
echo UNet 分割服务启动中...
echo ============================================================
cd /d D:\shengwu\using\unet-pytorch
python api_server.py
