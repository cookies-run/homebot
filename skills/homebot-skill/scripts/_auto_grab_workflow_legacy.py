#!/usr/bin/env python3
"""
Auto Grab Workflow - 兼容入口（已弃用）

此文件保留为旧入口。所有自主抓取逻辑已迁移到 grab_optimized.py。
当 Agent 或旧命令调用 auto_grab_workflow.py 时，会自动转发到 grab_optimized.py，
确保始终执行最新的五阶段状态机抓取流程。

用法（与之前完全兼容）:
    python auto_grab_workflow.py              # 默认抓取纸巾
    python auto_grab_workflow.py --target "一瓶矿泉水"
"""

import os
import sys
import warnings

# 让当前 scripts 目录可作为模块根，与 grab_optimized.py 自身导入方式一致
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 标记旧入口已弃用，但仍可继续工作
warnings.warn(
    "auto_grab_workflow.py 已弃用，将由 grab_optimized.py 接管。"
    "请优先使用 scripts/grab_optimized.py 或 MCP auto_grab 工具。",
    DeprecationWarning,
    stacklevel=2,
)

# 直接转发到 grab_optimized.py 的 main()
# 这样旧引用、旧命令、旧 Agent 调用都能无感切换到优化版实现
import grab_optimized

if __name__ == "__main__":
    print("[INFO] auto_grab_workflow.py -> 转发到 grab_optimized.py")
    grab_optimized.main()
