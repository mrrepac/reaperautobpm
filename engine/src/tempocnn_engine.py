"""TempoCNN-based BPM engine for the "Detect beat BPM" REAPER action.

Feature extraction (read_features/_ensure_length/_to_sliding_window) and
normalization (std_normalizer) below are ported from tempo-cnn
(https://github.com/hendriks73/tempo-cnn), Copyright Hendrik Schreiber,
licensed under AGPLv3 (https://www.gnu.org/licenses/agpl-3.0.html).
The model file deeptemp_k16.onnx is a format conversion of that project's
deeptemp_k16.h5, same license.

The mel spectrogram is computed here in numpy rather than through librosa.
librosa pulls in numba, scipy and joblib, which tripled the frozen build and
cost seconds of import time on every run; the front end is a fixed pipeline of
four operations, so it is spelled out instead. Audio decoding and resampling
still go through soundfile and soxr, which is what librosa itself calls, so
the samples reaching the model are bit-identical -- see
dev/model_conversion/verify_parity.py.

Usage:
    tempocnn_engine.exe --job <job_file>
    tempocnn_engine.exe <audio_path> [--offset S] [--duration S]
                                     [--snap-length S]

Prints `key: value` lines to stdout on success (exit 0); `bpm` is the answer
and `origin` says how it was reached. On failure, prints an error message to
stderr and exits non-zero.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

# Front end. Fixed by the trained model: these are the values tempo-cnn used.
SR = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
WINDOW_FRAMES = 256
WINDOW_HOP = 128
BPM_OFFSET = 30

# Fine tempo refinement.
REFINE_BAND = 0.06      # searched either side of the network's estimate
REFINE_STEP = 0.02      # BPM
REFINE_HARMONICS = (1.0, 0.5, 1.0 / 3.0, 0.25)
REFINE_MIN_PROMINENCE = 1.25   # peak over median score, or the band is flat
REFINE_MIN_DOMINANCE = 1.15    # peak over the best rival outside its main lobe
MIN_REFINE_SECONDS = 4.0

# Loop snapping. A trimmed-to-the-bar item pins the tempo exactly, but only if
# the fit is tight: with a loose tolerance an arbitrary trim also "fits".
SNAP_BEAT_GROUP = 4     # whole 4/4 bars only
SNAP_TOLERANCE_BEATS = 0.05
SNAP_MIN_SECONDS = 1.0

MIN_BPM = 20.0
MAX_BPM = 640.0


# Audio ------------------------------------------------------------------------


def load_audio(path: str, offset: float = 0.0, duration: float | None = None,
               wrap: bool = False
               ) -> np.ndarray:
    """Mono float32 at SR, mirroring librosa.load's soundfile path exactly.

    Same seek-then-read as librosa (so `offset`/`duration` land on the same
    samples), same mean-of-channels downmix, same soxr quality setting.
    """
    with sf.SoundFile(path) as descriptor:
        native_sr = descriptor.samplerate
        start = int(offset * native_sr)
        if wrap and descriptor.frames:
            start %= descriptor.frames
        if start:
            descriptor.seek(start)
        frames = int(duration * native_sr) if duration is not None else -1
        if wrap and frames >= 0:
            frames = min(frames, descriptor.frames)
        samples = descriptor.read(frames=frames, dtype="float32", always_2d=False)
        if wrap and frames > len(samples):
            descriptor.seek(0)
            head = descriptor.read(frames - len(samples), dtype="float32", always_2d=False)
            samples = np.concatenate((samples, head))
        samples = samples.T

    if samples.ndim > 1:
        samples = np.mean(samples, axis=tuple(range(samples.ndim - 1)))
    if samples.size == 0:
        raise ValueError("selected audio region is empty")

    if native_sr != SR:
        samples = soxr.resample(samples, native_sr, SR, quality="soxr_hq")
    return np.ascontiguousarray(samples, dtype=np.float32)


# Mel spectrogram --------------------------------------------------------------
#
# Reimplements librosa.feature.melspectrogram(power=1) with a Slaney-scale,
# Slaney-normalized filter bank and a periodic Hann window -- librosa's
# defaults. The float64 window against float32 frames, and the complex64 cast
# after the transform, are librosa's dtype choices and are kept so the result
# matches bit for bit rather than merely closely.


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(
        frequencies[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    frequencies = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(
        logstep * (mels[log_region] - min_log_mel))
    return frequencies


def mel_filterbank() -> np.ndarray:
    fft_frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    edges = _mel_to_hz(np.linspace(
        _hz_to_mel(np.array([FMIN]))[0],
        _hz_to_mel(np.array([FMAX]))[0],
        N_MELS + 2))

    weights = np.zeros((N_MELS, 1 + N_FFT // 2), dtype=np.float32)
    spacing = np.diff(edges)
    ramps = np.subtract.outer(edges, fft_frequencies)
    for index in range(N_MELS):
        lower = -ramps[index] / spacing[index]
        upper = ramps[index + 2] / spacing[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal area per filter, not equal peak.
    area = 2.0 / (edges[2:N_MELS + 2] - edges[:N_MELS])
    weights *= area[:, np.newaxis]
    return weights


def _periodic_hann(length: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def mel_spectrogram(samples: np.ndarray) -> np.ndarray:
    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="constant")
    frame_count = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    if frame_count < 1:
        raise ValueError("selected audio region is too short to analyse")

    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(N_FFT, frame_count),
        strides=(padded.strides[-1], HOP_LENGTH * padded.strides[-1]),
    )
    spectrum = np.fft.rfft(
        _periodic_hann(N_FFT)[:, np.newaxis] * frames, axis=0
    ).astype(np.complex64)
    magnitude = np.abs(spectrum)
    return mel_filterbank() @ magnitude


# Network input ----------------------------------------------------------------


def read_features(path: str, offset: float = 0.0,
                  duration: float | None = None,
                  wrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns the sliding-window network input and the raw mel spectrogram."""
    samples = load_audio(path, offset, duration, wrap)
    if not np.any(samples):
        raise ValueError("selected audio region is silent")
    spectrogram = mel_spectrogram(samples)
    data = np.reshape(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1], 1))
    if data.shape[2] < WINDOW_FRAMES:
        data = _ensure_length(data, WINDOW_FRAMES)
    return _to_sliding_window(data, WINDOW_FRAMES, WINDOW_HOP), spectrogram


def _ensure_length(data: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((1, data.shape[1], length, 1), dtype=data.dtype)
    padded[0, :, 0:data.shape[2], 0] = data[0, :, :, 0]
    return padded


def _to_sliding_window(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    total_frames = data.shape[2]
    windows = []
    span = ((total_frames - window_length) // hop_length + 1) * hop_length
    for offset in range(0, span, hop_length):
        windows.append(np.copy(data[:, :, offset:window_length + offset, :]))
    return np.concatenate(windows, axis=0)


def std_normalizer(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    if std != 0.0:
        data = (data - mean) / std
    return data.astype(np.float32)


def quad_interpol_argmax(y: np.ndarray, x: int | None = None) -> tuple[float, float]:
    if x is None:
        x = int(np.argmax(y))
    if x == 0 or x == y.shape[0] - 1:
        return float(x), float(y[x])
    z = np.polyfit([x - 1, x, x + 1], [y[x - 1], y[x], y[x + 1]], 2)
    argmax = -z[1] / (2.0 * z[0])
    height = z[2] - (z[1] ** 2.0) / (4.0 * z[0])
    return float(argmax), float(height)


def model_path() -> Path:
    """The model sits next to the executable, or one level up.

    A --onedir build puts the exe in its own folder inside engine/, while a
    source run starts from engine/src/; the model stays in engine/ either way.
    """
    roots = []
    for start in (Path(sys.argv[0]).resolve().parent, Path(__file__).resolve().parent):
        roots.extend((start, start.parent))
    for root in roots:
        candidate = root / "deeptemp_k16.onnx"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "deeptemp_k16.onnx not found next to the engine: " + str(roots[0]))


# Metric level -----------------------------------------------------------------


def distribution_candidates(averaged: np.ndarray, limit: int = 3
                            ) -> list[tuple[float, float]]:
    """Local peaks of the tempo distribution as (bpm, mass), strongest first.

    The network's own alternatives are worth more than arithmetic: when it is
    torn between 112 and 225 the runner-up carries real probability mass, and
    when it is sure the list is one entry long.
    """
    interior = np.arange(1, averaged.shape[0] - 1)
    peaks = interior[(averaged[1:-1] >= averaged[:-2])
                     & (averaged[1:-1] > averaged[2:])]
    peaks = sorted(peaks, key=lambda index: averaged[index], reverse=True)

    candidates = []
    for index in peaks[:limit]:
        position, _ = quad_interpol_argmax(averaged, int(index))
        mass = float(np.sum(averaged[max(0, index - 1):index + 2]))
        candidates.append((position + BPM_OFFSET, mass))
    return candidates


def window_agreement(prediction: np.ndarray, bpm: float) -> float:
    """Share of windows hearing the same tempo, within 4%.

    Averaging the distribution hides disagreement: a file whose halves sit at
    different tempos produces a confident-looking average of two answers, and
    a single project tempo is the wrong answer for it.
    """
    if prediction.shape[0] < 2 or bpm <= 0:
        return 1.0
    per_window = np.argmax(prediction, axis=1) + BPM_OFFSET
    return float(np.mean(np.abs(per_window - bpm) / bpm <= 0.04))


# Fine refinement --------------------------------------------------------------


def onset_envelope(spectrogram: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux over the mel bands, mean-removed.

    Tapered, because the comb below reads this as a sum of sinusoids and an
    untapered edge leaks across the whole band.
    """
    decibels = 20.0 * np.log10(np.maximum(spectrogram, 1e-10))
    flux = np.maximum(0.0, np.diff(decibels, axis=1)).sum(axis=0)
    if flux.size < 8:
        return np.zeros(0)
    flux = flux - flux.mean()
    return flux * _periodic_hann(flux.size)


def comb_scores(envelope: np.ndarray, bpms: np.ndarray, fps: float) -> np.ndarray:
    """Harmonic comb response of the envelope at each candidate tempo.

    A plain DFT bin at the beat frequency resolves tempo no better than
    1/duration; summing the first four harmonics sharpens that fourfold, which
    is what makes a half-BPM answer meaningful rather than decorative.
    """
    frame_index = np.arange(envelope.size, dtype=np.float64)
    scores = np.zeros(bpms.size, dtype=np.float64)
    scale = np.sum(np.abs(envelope)) or 1.0

    for weight_index, weight in enumerate(REFINE_HARMONICS, start=1):
        cycles_per_frame = weight_index * bpms / 60.0 / fps
        usable = cycles_per_frame < 0.5     # above the envelope's own Nyquist
        if not np.any(usable):
            break
        # Chunked: the full exponential matrix would be hundreds of megabytes.
        for start in range(0, bpms.size, 128):
            block = slice(start, min(start + 128, bpms.size))
            active = usable[block]
            if not np.any(active):
                continue
            phase = np.outer(cycles_per_frame[block], frame_index)
            response = np.abs(np.exp(-2j * np.pi * phase) @ envelope)
            scores[block] += weight * np.where(active, response, 0.0) / scale
    return scores


def refine_bpm(spectrogram: np.ndarray, bpm: float) -> tuple[float, float] | None:
    """Sharpens the network's estimate against the onset envelope.

    Searched only within a narrow band, so the metric level the network chose
    is preserved -- this answers "112.4 or 112.5", never "112 or 225".
    """
    envelope = onset_envelope(spectrogram)
    fps = SR / HOP_LENGTH
    if envelope.size / fps < MIN_REFINE_SECONDS:
        return None

    low = max(MIN_BPM, bpm * (1.0 - REFINE_BAND))
    high = min(MAX_BPM, bpm * (1.0 + REFINE_BAND))
    if high <= low:
        return None

    bpms = np.arange(low, high + REFINE_STEP, REFINE_STEP)
    scores = comb_scores(envelope, bpms, fps)
    if not np.any(scores > 0):
        return None

    median = float(np.median(scores))
    peak = int(np.argmax(scores))
    prominence = float(scores[peak] / median) if median > 0 else 0.0
    if prominence < REFINE_MIN_PROMINENCE:
        return None

    # A single dominant peak means one periodicity; several comparable ones mean
    # the envelope is patterned rather than pulsed -- a loop seam repeating
    # every few seconds, say -- and then the peak the comb likes best is not
    # the tempo. Rivals within the peak's own main lobe (width 60/duration, the
    # resolution limit) are the same peak and do not count.
    lobe_bpm = 60.0 * fps / envelope.size
    lobe_steps = max(1, int(np.ceil(lobe_bpm / REFINE_STEP)))
    outside = np.ones(scores.size, dtype=bool)
    outside[max(0, peak - lobe_steps):peak + lobe_steps + 1] = False
    if np.any(outside):
        rival = float(np.max(scores[outside]))
        if rival > 0 and scores[peak] / rival < REFINE_MIN_DOMINANCE:
            return None

    position, _ = quad_interpol_argmax(scores, peak)
    refined = float(low + position * REFINE_STEP)
    if not low <= refined <= high:
        return None
    return refined, prominence


# Loop snapping ----------------------------------------------------------------


def snap_to_loop(bpm: float, length_seconds: float | None
                 ) -> tuple[float, int] | None:
    """Exact tempo for an item trimmed to whole bars, or None.

    Only whole 4/4 bars and only a very tight fit: at a loose tolerance an
    arbitrary trim lands on "a whole number of beats" by chance more often
    than not, and a confidently wrong exact answer is worse than no answer.
    """
    if not length_seconds or length_seconds < SNAP_MIN_SECONDS or bpm <= 0:
        return None
    beats = length_seconds * bpm / 60.0
    snapped = round(beats / SNAP_BEAT_GROUP) * SNAP_BEAT_GROUP
    if snapped < SNAP_BEAT_GROUP:
        return None
    if abs(snapped - beats) > SNAP_TOLERANCE_BEATS:
        return None
    return snapped * 60.0 / length_seconds, int(snapped)


# Estimation -------------------------------------------------------------------


def estimate(audio_path: str, offset: float = 0.0, duration: float | None = None,
             snap_length: float | None = None, wrap: bool = False) -> dict:
    windows, spectrogram = read_features(audio_path, offset, duration, wrap)

    session = ort.InferenceSession(str(model_path()),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: std_normalizer(windows)})[0]

    averaged = np.average(prediction, axis=0)
    index, height = quad_interpol_argmax(averaged)
    network_bpm = index + BPM_OFFSET

    result = {
        "bpm": network_bpm,
        "bpm_cnn": network_bpm,
        "confidence": max(0.0, min(1.0, height)),
        "origin": "cnn",
        "agreement": window_agreement(prediction, network_bpm),
        "candidates": distribution_candidates(averaged),
    }

    refined = refine_bpm(spectrogram, network_bpm)
    if refined is not None:
        result["bpm"] = refined[0]
        result["prominence"] = refined[1]
        result["origin"] = "refined"

    snapped = snap_to_loop(result["bpm"], snap_length)
    if snapped is not None:
        result["bpm"] = snapped[0]
        result["loop_beats"] = snapped[1]
        result["origin"] = "loop"

    return result


# Job file ---------------------------------------------------------------------


def parse_job(path: str) -> dict:
    """`key=value` lines in UTF-8. The value is the rest of the line: a path
    may hold anything, including an equals sign."""
    job = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            job[key.strip()] = value
    return job


def parse_arguments(argv: list[str]) -> dict:
    if not argv:
        raise ValueError("usage: tempocnn_engine.exe --job <job_file>")

    if argv[0] == "--job":
        if len(argv) != 2:
            raise ValueError("--job takes exactly one path")
        return parse_job(argv[1])

    job = {"path": argv[0]}
    flags = {"--offset": "offset", "--duration": "duration",
             "--snap-length": "snap_length"}
    index = 1
    while index < len(argv):
        if argv[index] not in flags or index + 1 >= len(argv):
            raise ValueError(f"unexpected argument: {argv[index]}")
        job[flags[argv[index]]] = argv[index + 1]
        index += 2
    return job


def _optional_float(job: dict, key: str) -> float | None:
    value = job.get(key)
    if value is None or value == "":
        return None
    number = float(value)
    return number if number > 0 else None


# Output -----------------------------------------------------------------------


def format_result(result: dict) -> str:
    lines = [
        f"bpm: {result['bpm']:.3f}",
        f"bpm_cnn: {result['bpm_cnn']:.3f}",
        f"confidence: {result['confidence']:.3f}",
        f"origin: {result['origin']}",
        f"agreement: {result['agreement']:.3f}",
    ]
    if "loop_beats" in result:
        lines.append(f"loop_beats: {result['loop_beats']}")
    if result["candidates"]:
        lines.append("candidates: " + " ".join(
            f"{bpm:.2f}:{mass:.3f}" for bpm, mass in result["candidates"]))
    return "\n".join(lines)


def main() -> int:
    # Paths echoed back in error messages are whatever the filesystem holds, so
    # the default locale encoding is not enough to print them. PYTHONIOENCODING
    # cannot do this for us: the frozen build runs an isolated interpreter
    # config that ignores the PYTHON* environment variables.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    try:
        job = parse_arguments(sys.argv[1:])
        audio_path = job.get("path", "")
        if not audio_path:
            raise ValueError("no audio path given")
        if not Path(audio_path).is_file():
            print(f"audio file not found: {audio_path}", file=sys.stderr)
            return 1
        result = estimate(
            audio_path,
            offset=_optional_float(job, "offset") or 0.0,
            duration=_optional_float(job, "duration"),
            snap_length=_optional_float(job, "snap_length"),
            wrap=job.get("wrap") == "1",
        )
    except sf.LibsndfileError as exc:
        print(f"the engine cannot decode this file ({exc}). Formats: WAV, "
              f"AIFF, FLAC, OGG, OPUS, MP3. Glue or render the item to WAV.",
              file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"tempocnn engine failed: {exc}", file=sys.stderr)
        return 1

    if not MIN_BPM <= result["bpm"] <= MAX_BPM:
        print(f"implausible tempo: {result['bpm']:.3f}", file=sys.stderr)
        return 1

    print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
