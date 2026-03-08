"""Tests for pool manager."""

import struct
from pathlib import Path

import pytest

from scropipe.pool_manager import PoolManager


def make_wav(path: Path, num_samples: int = 100) -> Path:
    """Create a minimal valid WAV file (PCM 16-bit mono 44100 Hz)."""
    num_channels = 1
    sample_rate = 44100
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    # 44-byte header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    silence = b"\x00" * data_size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + silence)
    return path


@pytest.fixture
def pools_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pools"
    d.mkdir()
    return d


@pytest.fixture
def pm(pools_dir: Path) -> PoolManager:
    return PoolManager(pools_dir)


class TestListPools:
    def test_list_pools_empty(self, pm: PoolManager) -> None:
        assert pm.list_pools() == []

    def test_list_pools_returns_created(self, pm: PoolManager) -> None:
        pm.create_pool("alpha")
        pm.create_pool("beta")
        names = [p.name for p in pm.list_pools()]
        assert sorted(names) == ["alpha", "beta"]


class TestCreatePool:
    def test_create_pool(self, pm: PoolManager) -> None:
        info = pm.create_pool("my-pool")
        assert info.name == "my-pool"
        assert info.sample_count == 0
        assert info.sources == []
        assert info.created_at  # non-empty timestamp

    def test_create_duplicate_raises(self, pm: PoolManager) -> None:
        pm.create_pool("dup")
        with pytest.raises(ValueError, match="dup"):
            pm.create_pool("dup")


class TestGetPool:
    def test_get_pool(self, pm: PoolManager) -> None:
        pm.create_pool("test")
        info = pm.get_pool("test")
        assert info.name == "test"

    def test_get_nonexistent_raises(self, pm: PoolManager) -> None:
        with pytest.raises(KeyError):
            pm.get_pool("nope")


class TestPoolSamplesDir:
    def test_pool_samples_dir(self, pm: PoolManager, pools_dir: Path) -> None:
        pm.create_pool("drums")
        samples_dir = pm.get_samples_dir("drums")
        assert samples_dir == pools_dir / "drums" / "samples"
        assert samples_dir.is_dir()


class TestAddFiles:
    def test_add_files_to_pool(self, pm: PoolManager, tmp_path: Path) -> None:
        pm.create_pool("wavs")
        src = tmp_path / "source"
        src.mkdir()
        wav1 = make_wav(src / "a.wav")
        wav2 = make_wav(src / "b.wav")
        count = pm.add_files("wavs", [wav1, wav2])
        assert count == 2
        info = pm.get_pool("wavs")
        assert info.sample_count == 2
        samples = list(pm.get_samples_dir("wavs").iterdir())
        assert len(samples) == 2

    def test_add_files_handles_duplicates(self, pm: PoolManager, tmp_path: Path) -> None:
        pm.create_pool("wavs")
        src = tmp_path / "source"
        src.mkdir()
        wav = make_wav(src / "a.wav")
        pm.add_files("wavs", [wav])
        count = pm.add_files("wavs", [wav])
        assert count == 0  # duplicate, not re-added
        info = pm.get_pool("wavs")
        assert info.sample_count == 1

    def test_add_files_records_source(self, pm: PoolManager, tmp_path: Path) -> None:
        pm.create_pool("wavs")
        src = tmp_path / "source"
        src.mkdir()
        wav = make_wav(src / "x.wav")
        pm.add_files("wavs", [wav])
        info = pm.get_pool("wavs")
        assert len(info.sources) == 1
        assert info.sources[0].source_type == "files"
        assert info.sources[0].count == 1


class TestAddDirectory:
    def test_add_directory_to_pool(self, pm: PoolManager, tmp_path: Path) -> None:
        pm.create_pool("dir-pool")
        src = tmp_path / "audio"
        make_wav(src / "one.wav")
        make_wav(src / "sub" / "two.wav")
        # also put a non-wav file
        (src / "readme.txt").write_text("not audio")
        count = pm.add_directory("dir-pool", src)
        assert count == 2
        info = pm.get_pool("dir-pool")
        assert info.sample_count == 2
        assert len(info.sources) == 1
        assert info.sources[0].source_type == "directory"


class TestDeletePool:
    def test_delete_pool(self, pm: PoolManager, pools_dir: Path) -> None:
        pm.create_pool("doomed")
        pm.delete_pool("doomed")
        assert not (pools_dir / "doomed").exists()
        assert pm.list_pools() == []

    def test_delete_nonexistent_raises(self, pm: PoolManager) -> None:
        with pytest.raises(KeyError):
            pm.delete_pool("ghost")
