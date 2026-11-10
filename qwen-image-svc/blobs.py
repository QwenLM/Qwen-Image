"""结果落盘：内容寻址 + TTL 回收。

职责边界：只认 bytes 和 hash，不认任务、不认 HTTP。
内容寻址顺带白拿了去重 —— 同 seed 同参数的重复请求只占一份磁盘。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path


class BlobStore:
    def __init__(self, root: str, ttl_seconds: int = 24 * 3600) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweeper.start()

    # ---------------------------------------------------------------- 写
    def put(self, data: bytes, ext: str = "png") -> str:
        """返回文件名（不是路径），对外 URL 用它拼。"""
        digest = hashlib.sha256(data).hexdigest()
        name = f"{digest}.{ext}"
        path = self._path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # 原子：并发同 hash 不会读到半截文件
        else:
            os.utime(path)  # 续命：被再次引用的 blob 不该被清掉
        return name

    def get(self, name: str) -> bytes:
        return self._path(name).read_bytes()

    # ---------------------------------------------------------------- 内部
    def _path(self, name: str) -> Path:
        # 两层分片，避免单目录塞几万个文件
        return self.root / name[:2] / name

    def _sweep_loop(self) -> None:
        while not self._stop.wait(600):
            cutoff = time.time() - self.ttl
            for shard in self.root.iterdir():
                if not shard.is_dir():
                    continue
                for f in shard.iterdir():
                    try:
                        if f.stat().st_mtime < cutoff:
                            f.unlink()
                    except OSError:
                        pass

    def close(self) -> None:
        self._stop.set()


def ensure_clean_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)