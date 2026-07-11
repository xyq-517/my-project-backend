# app.py - Railway 部署入口文件
#
# 为什么要有这个文件？
# Railway Nixpacks 会自动检测并启动 app.py，所以用它作为入口最稳妥。
#
# 这个文件做三件事：
# 1. 从 COS 下载数据（如果配置了环境变量）
# 2. 导入 api_web_final 的 Flask app（实际的 API 逻辑）
# 3. 生产环境用 gunicorn 启动，本地开发用 Flask dev server
#
# 注意：这个文件里不要导入 unet / cv2 / torch 等重型依赖，
#       预生成模式不需要它们，避免导入失败。

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
    print(f"[警告] 数据下载失败（将尝试使用本地数据）: {e}")
    import traceback
    traceback.print_exc()

# 第二步：导入 API 服务（api_web_final.py）
# 注意：api_web_final 在模块加载时会执行初始化
print("正在加载 API 服务...")
from api_web_final import app
print("API 服务加载完成")
print()


def run_production():
    """生产环境：用 gunicorn 启动"""
    port = os.environ.get('PORT', '8080')
    bind_addr = f"0.0.0.0:{port}"

    print("=" * 60)
    print("  生产模式启动 (gunicorn)")
    print(f"  监听地址: {bind_addr}")
    print(f"  健康检查: GET /api/health")
    print(f"  分割接口: POST /api/segment")
    print("=" * 60)
    print()

    try:
        from gunicorn.app.wsgiapp import WSGIApplication

        sys.argv = [
            'gunicorn',
            'app:app',
            '--bind', bind_addr,
            '--workers', '1',
            '--timeout', '120',
            '--preload',
        ]
        WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]").run()
    except ImportError:
        # 没有 gunicorn 的话，降级用 Flask dev server
        print("[警告] gunicorn 未安装，降级使用 Flask 开发服务器")
        app.run(host='0.0.0.0', port=int(port), debug=False, threaded=True)


def run_development():
    """本地开发：用 Flask 开发服务器"""
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(f"  开发模式启动")
    print(f"  访问地址: http://0.0.0.0:{port}")
    print(f"  健康检查: http://0.0.0.0:{port}/api/health")
    print(f"  分割接口: POST http://0.0.0.0:{port}/api/segment")
    print("=" * 60)
    print()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    # 判断是不是生产环境（Railway 上会有 RAILWAY_SERVICE_NAME 环境变量）
    is_production = os.environ.get('RAILWAY_SERVICE_NAME') or os.environ.get('RAILWAY_ENVIRONMENT')

    if is_production:
        run_production()
    else:
        run_development()
