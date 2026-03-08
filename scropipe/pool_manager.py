"""Pool manager for organizing sample collections."""

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PoolSource:
    """Record of how samples were added to a pool."""

    source_type: str  # "files", "directory", "split"
    path: str
    count: int
    added_at: str


@dataclass
class PoolInfo:
    """Metadata about a sample pool."""

    name: str
    created_at: str
    sample_count: int = 0
    sources: list[PoolSource] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PoolManager:
    """Manages sample pools on disk.

    Pool structure:
        pools_dir/
          pool-name/
            pool.json    # PoolInfo serialized
            samples/     # WAV files
    """

    def __init__(self, pools_dir: Path) -> None:
        self.pools_dir = Path(pools_dir)
        self.pools_dir.mkdir(parents=True, exist_ok=True)

    def _pool_dir(self, name: str) -> Path:
        return self.pools_dir / name

    def _pool_json(self, name: str) -> Path:
        return self._pool_dir(name) / "pool.json"

    def _samples_dir(self, name: str) -> Path:
        return self._pool_dir(name) / "samples"

    def _save(self, info: PoolInfo) -> None:
        """Write PoolInfo to disk as JSON."""
        data = asdict(info)
        self._pool_json(info.name).write_text(json.dumps(data, indent=2))

    def _load(self, name: str) -> PoolInfo:
        """Read PoolInfo from disk."""
        data = json.loads(self._pool_json(name).read_text())
        sources = [PoolSource(**s) for s in data.pop("sources", [])]
        return PoolInfo(**data, sources=sources)

    def list_pools(self) -> list[PoolInfo]:
        """List all pools."""
        pools: list[PoolInfo] = []
        for d in sorted(self.pools_dir.iterdir()):
            if d.is_dir() and (d / "pool.json").exists():
                pools.append(self._load(d.name))
        return pools

    def create_pool(self, name: str) -> PoolInfo:
        """Create a new pool.

        Raises:
            ValueError: If a pool with this name already exists.
        """
        if self._pool_dir(name).exists():
            raise ValueError(f"Pool already exists: {name}")
        self._pool_dir(name).mkdir(parents=True)
        self._samples_dir(name).mkdir()
        info = PoolInfo(name=name, created_at=_now_iso())
        self._save(info)
        return info

    def get_pool(self, name: str) -> PoolInfo:
        """Get pool metadata.

        Raises:
            KeyError: If the pool does not exist.
        """
        if not self._pool_json(name).exists():
            raise KeyError(f"Pool not found: {name}")
        return self._load(name)

    def get_samples_dir(self, name: str) -> Path:
        """Get the samples directory for a pool, ensuring it exists."""
        samples = self._samples_dir(name)
        samples.mkdir(parents=True, exist_ok=True)
        return samples

    def add_files(self, name: str, files: list[Path]) -> int:
        """Copy WAV files into a pool.

        Handles duplicates by skipping files whose name already exists in the
        pool's samples directory.

        Args:
            name: Pool name.
            files: List of WAV file paths to add.

        Returns:
            Number of files actually added (excluding duplicates).
        """
        info = self.get_pool(name)
        samples_dir = self.get_samples_dir(name)
        added = 0
        for f in files:
            dest = samples_dir / f.name
            if dest.exists():
                continue
            shutil.copy2(f, dest)
            added += 1
        info.sample_count += added
        info.sources.append(
            PoolSource(
                source_type="files",
                path=str(files[0].parent) if files else "",
                count=added,
                added_at=_now_iso(),
            )
        )
        self._save(info)
        return added

    def add_directory(self, name: str, directory: Path) -> int:
        """Copy all WAV files from a directory (recursively) into a pool.

        Args:
            name: Pool name.
            directory: Directory to scan for WAV files.

        Returns:
            Number of files added.
        """
        wavs = sorted(Path(directory).rglob("*.wav"))
        info = self.get_pool(name)
        samples_dir = self.get_samples_dir(name)
        added = 0
        for f in wavs:
            dest = samples_dir / f.name
            if dest.exists():
                continue
            shutil.copy2(f, dest)
            added += 1
        info.sample_count += added
        info.sources.append(
            PoolSource(
                source_type="directory",
                path=str(directory),
                count=added,
                added_at=_now_iso(),
            )
        )
        self._save(info)
        return added

    def delete_pool(self, name: str) -> None:
        """Remove an entire pool directory.

        Raises:
            KeyError: If the pool does not exist.
        """
        pool_dir = self._pool_dir(name)
        if not pool_dir.exists():
            raise KeyError(f"Pool not found: {name}")
        shutil.rmtree(pool_dir)
