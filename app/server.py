"""
Cardiac Agent 系统 - 服务入口

启动方式:
  python -m app.server --serve              # 启动 API 服务 (默认端口 8005)
  python -m app.server --serve --port 8080  # 指定端口
  python -m app.server --check              # 检查服务状态

系统架构:
  用户上传 → Agent序列识别 → 智能抽帧 → 模态排序 → Agent决定API → Expert Worker → Agent总结
"""

import argparse
import os
import sys

# server.py 位于 app/ 内，需要将 MMedAgent 根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Cardiac Agent System")
    parser.add_argument("--check", action="store_true", help="检查服务状态")
    parser.add_argument("--serve", action="store_true", help="启动API服务")
    parser.add_argument("--port", type=int, default=8005, help="API服务端口")

    args = parser.parse_args()

    if args.check:
        from tests.test_examples import test_service_status
        test_service_status()
    elif args.serve:
        import uvicorn
        from app.routes.api import create_api_app

        app = create_api_app()
        print(f"\n🚀 启动 Cardiac Agent API 服务: http://0.0.0.0:{args.port}")
        print(f"📖 API 文档: http://0.0.0.0:{args.port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        from tests.test_examples import test_service_status
        test_service_status()


if __name__ == "__main__":
    main()
