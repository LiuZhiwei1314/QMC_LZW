import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import ml_collections
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tools.utils import system


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, ml_collections.ConfigDict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, system.Atom):
        return {
            'symbol': value.symbol,
            'coords': list(value.coords),
            'charge': float(value.charge),
            'atomic_number': int(value.atomic_number),
            'units': value.units,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class MetricsLogger:
    def __init__(self, run_dir: Path, enable_tensorboard: bool = True):
        self.logs_dir = run_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.logs_dir / 'metrics.csv'
        self.jsonl_path = self.logs_dir / 'metrics.jsonl'
        self._tb_writer = None
        if enable_tensorboard:
            tensorboard_dir = run_dir / 'tensorboard'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = EventFileWriter(str(tensorboard_dir))
        if not self.csv_path.exists():
            with self.csv_path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['timestamp', 'stage', 'step', 'metric', 'value'])

    def log_scalars(self, stage: str, step: int, scalars: Mapping[str, float]) -> None:
        timestamp = time.time()
        with self.csv_path.open('a', newline='') as handle:
            writer = csv.writer(handle)
            for name, value in scalars.items():
                writer.writerow([timestamp, stage, step, name, float(value)])

        record = {
            'timestamp': timestamp,
            'stage': stage,
            'step': step,
            'metrics': {name: float(value) for name, value in scalars.items()},
        }
        with self.jsonl_path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')

        if self._tb_writer is not None:
            for name, value in scalars.items():
                event = Event(
                    wall_time=timestamp,
                    step=step,
                    summary=Summary(
                        value=[Summary.Value(tag=f'{stage}/{name}', simple_value=float(value))]
                    ),
                )
                self._tb_writer.add_event(event)
            self._tb_writer.flush()

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()


class CheckpointManager:
    def __init__(self, run_dir: Path):
        self.checkpoint_dir = run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.checkpoint_dir / 'last.pkl'

    def _write_pickle(self, path: Path, payload: Mapping[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        with tmp_path.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def save_last(self, payload: Mapping[str, Any]) -> None:
        self._write_pickle(self.last_path, payload)

    def save_step(self, stage: str, step: int, payload: Mapping[str, Any]) -> Path:
        path = self.checkpoint_dir / f'{stage}_step_{step:08d}.pkl'
        self._write_pickle(path, payload)
        self.save_last(payload)
        return path

    def load_last(self) -> Optional[Mapping[str, Any]]:
        if not self.last_path.exists():
            return None
        with self.last_path.open('rb') as handle:
            return pickle.load(handle)


class RunManager:
    def __init__(self, output_cfg: ml_collections.ConfigDict):
        self.run_dir = Path(output_cfg.root_dir).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = int(output_cfg.checkpoint_every)
        self.metrics_every = int(output_cfg.metrics_every)
        self.resume = bool(output_cfg.resume)
        self.enable_tensorboard = bool(output_cfg.enable_tensorboard)
        self.logger = MetricsLogger(self.run_dir, enable_tensorboard=self.enable_tensorboard)
        self.checkpoints = CheckpointManager(self.run_dir)

    def save_config(self, cfg: ml_collections.ConfigDict) -> None:
        config_path = self.run_dir / 'config.json'
        config_path.write_text(json.dumps(_to_jsonable(cfg), indent=2, sort_keys=True))

    def should_log(self, step: int, total_steps: int) -> bool:
        return step == total_steps or step % self.metrics_every == 0

    def should_checkpoint(self, step: int, total_steps: int) -> bool:
        return step == total_steps or step % self.checkpoint_every == 0

    def log_scalars(self, stage: str, step: int, scalars: Mapping[str, float]) -> None:
        self.logger.log_scalars(stage, step, scalars)

    def load_last_checkpoint(self) -> Optional[Mapping[str, Any]]:
        if not self.resume:
            return None
        return self.checkpoints.load_last()

    def close(self) -> None:
        self.logger.close()
