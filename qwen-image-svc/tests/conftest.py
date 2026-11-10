import sys
from pathlib import Path

# api/ 是一层包，业务模块在仓库根目录平铺；测试也这样导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))