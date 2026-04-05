# FlowDec wrapper: loads NDAC + FlowDec, provides encode/decode, and SR conversion utilities
import os
import importlib.util, sys
import torch

# codec wrapper abstract
class Codec:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FlowDecWrapper(Codec):
    """
    Wraps FlowDec (enhancer) with NDAC (codec backbone) for simple encode/decode.

    Contract
    - Inputs/outputs are torch.Tensors with shape [batch, channels, time].
    - encode(wv_44100): expects 44.1kHz audio. Internally upsamples to 48kHz for NDAC.
      Returns quantized latent zq (as produced by NDAC quantizer.from_codes).
    - decode(zq): decodes zq to waveform with NDAC, enhances with FlowDec,
      downsamples back to 44.1kHz, returns waveform tensor in [-1, 1].
    """

    def __init__(
        self,
        ckpt_dir: str = "models/FlowDec/checkpoints",
        model_name: str = "flowdec_25s",
        ndac_model: str = "ndac-25",
        n_quantizers: int = 16,
        enhance_steps: int = 3,
        enhance_solver: str = "midpoint",
        device: str | None = None,
        flowdec_repo_dir: str = "models/FlowDec",
    ) -> None:
        # Pick device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.ndac_model = ndac_model
        self.n_quantizers = n_quantizers
        self.enhance_steps = enhance_steps
        self.enhance_solver = enhance_solver
        self.flowdec_repo_dir = flowdec_repo_dir

        # Ensure FlowDec repo is importable (for dac, configs, etc.)
        if flowdec_repo_dir not in sys.path:
            sys.path.append(flowdec_repo_dir)

        # Lazy imports that depend on FlowDec repo
        from dac import DAC  # NDAC backbone
        from hydra import compose, initialize_config_dir, initialize
        from hydra.utils import instantiate
        from hydra.core.global_hydra import GlobalHydra

        # 1) Load NDAC weights
        dac_weights = os.path.join(
            ckpt_dir, f"ndac/{ndac_model}/800k/dac/weights.pth"
        )
        if not os.path.isfile(dac_weights):
            raise FileNotFoundError(
                f"NDAC weights not found at: {dac_weights}. Did you download checkpoints?"
            )
        self.dac_model = DAC.load(dac_weights)
        self.dac_model.to(self.device).eval()

        # 2) Initialize Hydra (config) and load FlowDec enhancer
        config_dir = os.path.join(flowdec_repo_dir, "config")
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"FlowDec config directory not found at: {config_dir}. Check flowdec_repo_dir."
            )
        # Initialize Hydra once per process
        if not GlobalHydra.instance().is_initialized():
            # initialize_config_dir(config_dir=config_dir, version_base="1.3")
            initialize(config_path=os.path.join('../',config_dir), version_base="1.3")
        conf = compose(config_name=model_name)

        ckpt_path = os.path.join(ckpt_dir, f"flowdec/{model_name}/step=800000.ckpt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"FlowDec checkpoint not found at: {ckpt_path}. Did you download checkpoints?"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_key = "_pl_ema_state_dict" if "_pl_ema_state_dict" in ckpt else "state_dict"

        self.model = instantiate(conf["model"])  # FlowDec enhancer
        self.model.load_state_dict(ckpt[state_key])
        self.model.to(torch.float32)
        self.model.to(self.device).eval()

    # --------- SR Utilities ---------
    @staticmethod
    def to_48000(x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected [batch, channels, time]"
        L = x.shape[-1]
        new_len = int(round(L * 48000 / 44100))
        orig_dtype = x.dtype
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
        y = torch.nn.functional.interpolate(
            x, size=new_len, mode="linear", align_corners=False
        )
        return y.to(orig_dtype)

    @staticmethod
    def to_44100(x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected [batch, channels, time]"
        L = x.shape[-1]
        new_len = int(round(L * 44100 / 48000))
        orig_dtype = x.dtype
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
        y = torch.nn.functional.interpolate(
            x, size=new_len, mode="linear", align_corners=False
        )
        return y.to(orig_dtype)

    # --------- Codec I/O ---------
    @torch.no_grad()
    def encode(self, wv_44100: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wv_44100: Tensor [B, C, T] at 44.1kHz
        Returns:
            zq: Quantized latent tensor suitable for NDAC decode
        """
        assert wv_44100.dim() == 3, "Expected input shape [batch, channels, time]"
        wv_44100 = wv_44100.to(self.device)
        wv_48000 = self.to_48000(wv_44100)

        # NDAC expects 48kHz input
        x = self.dac_model.preprocess(wv_48000, 48000)
        z, codes, latents, _, _ = self.dac_model.encode(
            x, n_quantizers=self.n_quantizers
        )
        zq, _, _ = self.dac_model.quantizer.from_codes(codes)
        return zq

    @torch.no_grad()
    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zq: Quantized latent from encode() or compatible tensor
        Returns:
            rec_44100: Reconstructed/enhanced waveform [B, C, T] at 44.1kHz
        """
        zq = zq.to(self.device)
        # NDAC decode
        xhat_ndac = self.dac_model.decode(zq)
        # FlowDec enhancement
        xhat_flowdec = self.model.enhance(
            xhat_ndac, N=self.enhance_steps, solver=self.enhance_solver
        )
        # Prevent clipping
        max_abs = xhat_flowdec.abs().max()
        if max_abs > 1.0:
            xhat_flowdec = xhat_flowdec / max_abs
        # Return at 44.1kHz
        rec_44100 = self.to_44100(xhat_flowdec)
        return rec_44100


class StableAudioOpenWrapper(Codec):
    """
    Wrapper for Stable Audio Open autoencoder with simple encode/decode API.

    - Loads the pretrained autoencoder via the streamable-stable-audio-open repo utilities.
    - encode(wv): returns latents from autoencoder.encode(wv)
    - decode(latents): returns reconstructed waveform from autoencoder.decode(latents)
    """

    def __init__(
        self,
        pretrained_id: str = "stabilityai/stable-audio-open-1.0",
        model_half: bool = False,
        skip_bottleneck: bool = True,
        device: str | None = None,
        repo_dir: str = "models/streamable-stable-audio-open",
        use_cached_conv: bool = False,
    ) -> None:
        # Pick device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Ensure the repo dir is importable and import required utilities
        # Insert at the front to avoid clashing with top-level "models" package
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        try:
            from export import remove_parametrizations  # type: ignore
            from models import get_pretrained_pretransform  # type: ignore
            import cached_conv as cc  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Failed to import Stable Audio Open utilities from '{repo_dir}'. "
                f"Make sure the repository exists and is intact. Original error: {e}"
            )

        cc.use_cached_conv(use_cached_conv)

        # Load pretrained autoencoder
        autoencoder, _model_config = get_pretrained_pretransform(
            pretrained_id,
            model_half=model_half,
            skip_bottleneck=skip_bottleneck,
            device=self.device,
        )

        # Remove parametrizations for inference
        remove_parametrizations(autoencoder)

        self.autoencoder = autoencoder.to(self.device).eval()

        sys.path.remove(repo_dir)

    @torch.no_grad()
    def encode(self, wv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wv: Tensor [B, C, T] waveform (expected 44.1kHz as per Stable Audio Open)
        Returns:
            latents: Autoencoder latent tensor
        """
        assert wv.dim() == 3, "Expected input shape [batch, channels, time]"
        wv = wv.to(self.device)
        latents = self.autoencoder.encode(wv)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: Latent tensor returned by encode()
        Returns:
            rec: Reconstructed waveform [B, C, T]
        """
        latents = latents.to(self.device)
        rec = self.autoencoder.decode(latents)
        # Prevent clipping if any op produced values outside [-1,1]
        max_abs = rec.abs().max()
        if max_abs > 1.0:
            rec = rec / max_abs
        return rec

def import_class_from_file(file_path: str, class_name: str, module_name: str = "dynamic_module"):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

class Music2LatentWrapper(Codec):
    """
    Wrapper for Music2Latent scripted UNet with simple encode/decode API.

    - Loads ScriptedUNet from the music2latent-scripted repository.
    - Applies required hparams tweaks (ratio/out_channels) before init.
    - Loads weights from the provided checkpoint path.
    - encode(wv): returns latent representation
    - decode(latent): reconstructs waveform
    """

    def __init__(
        self,
        repo_root: str = "models/music2latent-scripted/music2latent",
        package_name: str = "music2latent",
        checkpoint_path: str = "models/music2latent.pt",
        device: str | None = None,
        ratio: int = 4096,
        out_channels: int = 1,
    ) -> None:
        # Pick device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Ensure repo root is importable, then import package modules
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            # Import via package to avoid path-specific import issues
            ScriptedUNet = import_class_from_file(f"{repo_root}/export.py", "ScriptedUNet")
            hparams = import_class_from_file(f"{repo_root}/hparams.py", "hparams")
            sigma_rescale = import_class_from_file(f"{repo_root}/hparams_inference.py", "sigma_rescale")

            # ScriptedUNet = getattr(export_mod, "ScriptedUNet")
            # hparams = getattr(hparams_mod, "hparams")
            # sigma_rescale = getattr(hinf_mod, "sigma_rescale")
        except Exception as e:
            raise ImportError(
                f"Failed to import Music2Latent from '{repo_root}/{package_name}'. "
                f"Ensure the repository is present. Original error: {e}"
            )

        # Tweak hparams as used in the notebook
        setattr(hparams, "ratio", ratio)
        setattr(hparams, "out_channels", out_channels)

        # Build model
        self.model = ScriptedUNet(hparams, sigma_rescale=sigma_rescale).to(self.device)

        # Load checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Music2Latent checkpoint not found at: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("gen_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        sys.path.remove(repo_root)

    @torch.no_grad()
    def encode(self, wv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wv: Tensor [B, C, T] waveform
        Returns:
            latent: Model latent representation
        """
        assert wv.dim() == 3, "Expected input shape [batch, channels, time]"
        wv = wv.to(self.device)
        latent = self.model.encode(wv)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent tensor returned by encode()
        Returns:
            rec: Reconstructed waveform [B, C, T]
        """
        latent = latent.to(self.device)
        rec = self.model.decode(latent)
        return rec
