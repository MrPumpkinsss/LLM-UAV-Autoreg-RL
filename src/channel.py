from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelState:
    positions: np.ndarray
    channel_gain: np.ndarray
    snr: np.ndarray
    pdp: np.ndarray


def sample_channel(cfg: dict, num_uavs: int, rng: np.random.Generator) -> ChannelState:
    area = float(cfg["area_m"])
    altitude = float(cfg["altitude_m"])
    xy = rng.uniform(0.0, area, size=(num_uavs, 2))
    positions = np.column_stack([xy, np.full(num_uavs, altitude)])

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.maximum(dist, 1.0)

    h0 = float(cfg["h0"])
    path_loss_exp = float(cfg["path_loss_exp"])
    mean_gain = h0 / np.power(dist, path_loss_exp)
    fading = rng.exponential(float(cfg["rayleigh_scale"]), size=(num_uavs, num_uavs))
    channel_gain = mean_gain * fading
    np.fill_diagonal(channel_gain, 0.0)

    tx_power = float(cfg["tx_power_w"])
    noise = float(cfg["noise_power_w"])
    snr = tx_power * channel_gain / noise
    np.fill_diagonal(snr, np.inf)

    gamma0 = float(cfg["snr_threshold"])
    pdp = np.exp(-np.maximum(snr, 0.0) / gamma0)
    pdp = np.clip(pdp, 0.0, 0.95)
    np.fill_diagonal(pdp, 0.0)

    return ChannelState(
        positions=positions.astype(np.float64),
        channel_gain=channel_gain.astype(np.float64),
        snr=snr.astype(np.float64),
        pdp=pdp.astype(np.float64),
    )
