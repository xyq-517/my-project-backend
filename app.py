# app.py - Railway 部署入口文件（最简稳定版）
#
# 为什么要有这个文件？
# Railway Nixpacks 会自动检测并启动 app.py，所以用它作为入口最稳妥。
#
# 这个文件做两件事：
# 1. 从 COS 下载数据（如果配置了环境变量）
# 2. 导入 api_web_final 的 Flask app，用 Flask 开发服务器启动
#
# 注意：
# - 用 Flask 开发服务器，简单稳定，不容易崩
# - 预生成模式不需要 unet/cv2/torch，避免导入失败
# - 生产环境也能用（并发量不大的话完全够用）

import os
import sys

# 确保工作目录正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("  胸腹部器官病灶检测平台 - API 服务")
print("=" * 60)
print(f"工作目录: {BASE_DIR}")
print()

# 第一步：下载数据（从 COS）
try:
    from data_loader import download_all_data
    download_all_data()
except Exception as e:
    print(f"[警告] 数据下载模块导入失败（将尝试使用本地数据）: {e}")

# 第二步：导入 API 服务（api_web_final.py）
print("正在加载 API 服务...")
from api_web_final import app
print("API 服务加载完成")
print()

# 第三步：启动服务
# 用 Flask 自带的开发服务器，简单稳定
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("=" * 60)
    print(f"  服务启动成功！")
    print(f"  监听地址: http://0.0.0.0:{port}")
    print(f"  健康检查: GET /api/health")
    print(f"  分割接口: POST /api/segment")
    print("=" * 60)
    print()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
