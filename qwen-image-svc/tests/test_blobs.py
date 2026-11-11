"""BlobStore 单测：纯文件系统，不涉及 GPU。"""
import time

from blobs import BlobStore


def test_put_is_content_addressed_and_deduped(tmp_path):
    store = BlobStore(str(tmp_path), ttl_seconds=3600)
    a = store.put(b"hello")
    b = store.put(b"hello")
    assert a == b
    assert store.get(a) == b"hello"
    assert len(list(tmp_path.rglob("*.png"))) == 1
    store.close()


def test_put_reattaches_mtime_on_hit(tmp_path):
    store = BlobStore(str(tmp_path), ttl_seconds=3600)
    name = store.put(b"x")
    path = store._path(name)
    old = time.time() - 100
    import os

    os.utime(path, (old, old))
    store.put(b"x")
    assert path.stat().st_mtime > old + 50
    store.close()


def test_sharded_layout(tmp_path):
    store = BlobStore(str(tmp_path), ttl_seconds=3600)
    name = store.put(b"y")
    assert name[:2] == __import__("hashlib").sha256(b"y").hexdigest()[:2]
    assert store._path(name).parent.name == name[:2]
    store.close()