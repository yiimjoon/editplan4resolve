"""Direct B-roll generator - replaces UVM subprocess integration."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from VideoForge.broll.comfyui_generator import ComfyUIGenerator
from VideoForge.broll.metadata import BrollMetadata, coerce_metadata_dict
from VideoForge.broll.stock_downloader import StockMediaDownloader, StockMediaResult

logger = logging.getLogger(__name__)


class DirectBrollGenerator:
    """
    Generate B-roll media directly using Stock APIs and ComfyUI.

    Replaces the UVM subprocess integration with native Python implementation.
    """

    def __init__(
        self,
        output_dir: Path,
        pexels_key: Optional[str] = None,
        unsplash_key: Optional[str] = None,
        comfyui_url: str = "http://127.0.0.1:8188",
        comfyui_workflow: Optional[Path] = None,
    ) -> None:
        """
        Initialize direct B-roll generator.

        Args:
            output_dir: Base output directory
            pexels_key: Pexels API key
            unsplash_key: Unsplash API key
            comfyui_url: ComfyUI server URL
            comfyui_workflow: Path to ComfyUI workflow JSON
        """
        self.output_dir = Path(output_dir)
        self.stock = StockMediaDownloader(pexels_key, unsplash_key)
        self.comfyui = ComfyUIGenerator(comfyui_url, comfyui_workflow)

    def generate_broll_batch(
        self,
        sentences: Iterable[Dict[str, object]],
        project_name: str,
        mode: str = "stock",
        config: Optional["BrollSourceConfig"] = None,
        timeout_sec: int = 600,
        on_progress: Optional[object] = None,
        scene_analyses: Optional[Iterable[Dict[str, object]]] = None,
    ) -> List[Path]:
        """
        Generate B-roll media for sentences.

        Args:
            sentences: List of sentence dicts with "text", "id", "uid"
            project_name: Project identifier
            mode: "stock", "ai", or "hybrid" (deprecated, overridden by config)
            config: Optional BrollSourceConfig for source/media selection
            timeout_sec: Max execution time per scene
            on_progress: Optional callback(dict) for progress updates
            scene_analyses: Optional LLM scene analysis results

        Returns:
            List of generated video file paths
        """
        if config is None:
            from VideoForge.broll.source_config import BrollSourceConfig

            config = BrollSourceConfig.from_config()

        source_mode = str(config.source_mode or "").strip()
        mode = self._resolve_source_mode(source_mode, mode)
        if mode is None:
            logger.info("Source mode is library_only; skipping generation.")
            return []

        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        prefer_video = config.prefer_media_type != "image_only"

        # Build scene map from analyses
        scene_map = {}
        if scene_analyses:
            scene_map = {s.get("sentence_id"): s for s in scene_analyses}

        sentence_list = list(sentences)
        generated_files = []
        total_scenes = len(sentence_list)

        try:
            for idx, sent in enumerate(sentence_list, start=1):
                if on_progress:
                    on_progress(
                        {
                            "stage": "Generating B-roll",
                            "current": idx,
                            "total": total_scenes,
                            "message": f"Scene {idx}/{total_scenes}",
                        }
                    )

                text = str(sent.get("text", "")).strip()
                sent_id = sent.get("id")
                uid = sent.get("metadata", {}).get("uid") or sent.get("uid", "")

                if not text:
                    logger.warning("Skipping empty sentence %d", idx)
                    continue

                # Get scene analysis if available
                scene = scene_map.get(sent_id, {})
                visual_tags = scene.get("visual_tags", [])
                scene_description = scene.get("scene_description", text)

                # Build search query from visual tags or text
                if visual_tags:
                    query = ", ".join(visual_tags[:3])
                else:
                    keywords = self._extract_search_tokens(text, limit=3)
                    query = ", ".join(keywords) if keywords else text

                logger.info("Scene %d: query='%s'", idx, query)

                # Generate media based on mode
                media_path = None
                media: Optional[StockMediaResult] = None

                if mode == "stock":
                    media = self._generate_stock(
                        query, project_dir, f"scene_{idx:03d}", scene, prefer_video=prefer_video
                    )
                elif mode == "ai":
                    media_path = self._generate_ai(
                        scene_description, project_dir, f"scene_{idx:03d}", scene
                    )
                elif mode == "hybrid":
                    # Try stock first, fallback to AI
                    media = self._generate_stock(
                        query, project_dir, f"scene_{idx:03d}", scene, prefer_video=prefer_video
                    )
                    if not media:
                        logger.info("Stock failed, trying AI fallback")
                        media_path = self._generate_ai(
                            scene_description, project_dir, f"scene_{idx:03d}", scene
                        )

                if media:
                    media_path = media.path
                if not media_path:
                    logger.warning("Failed to generate media for scene %d", idx)
                    continue

                # Build metadata
                if media:
                    meta = media.metadata
                else:
                    width = scene.get("comfyui_width") or 1920
                    height = scene.get("comfyui_height") or 1024
                    try:
                        width = int(width)
                        height = int(height)
                    except (TypeError, ValueError):
                        width, height = 1920, 1024
                    # AI mode: build metadata from ComfyUI generation
                    meta = self.comfyui.build_metadata(
                        positive_prompt=scene.get("comfyui_positive", scene_description),
                        negative_prompt=scene.get("comfyui_negative", ""),
                        seed=scene.get("comfyui_seed"),
                        width=width,
                        height=height,
                    )

                meta.apply_context(project_name, scene_description)
                if visual_tags:
                    meta.merge_keywords(visual_tags)
                if not meta.description:
                    meta.description = scene_description

                # Convert to video if it's an image
                if media_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    if not self._wait_for_file_ready(media_path):
                        logger.warning(
                            "Image not ready for conversion: %s", media_path.name
                        )
                        continue
                    video_path = self._image_to_video(
                        media_path, duration=5.0, output_name=f"scene_{idx:03d}.mp4"
                    )
                    if video_path:
                        meta.duration = meta.duration or 5.0
                        meta.raw_metadata.setdefault("scene_num", str(idx))
                        meta.raw_metadata.setdefault("uid", uid)
                        final_path = self._embed_metadata(video_path, meta.to_dict())
                        generated_files.append(final_path)
                    else:
                        logger.warning(
                            "Image-to-video conversion failed: %s", media_path.name
                        )
                else:
                    # Already a video
                    meta.raw_metadata.setdefault("scene_num", str(idx))
                    meta.raw_metadata.setdefault("uid", uid)
                    final_path = self._embed_metadata(media_path, meta.to_dict())
                    generated_files.append(final_path)
        finally:
            self.comfyui.stop_server()

        logger.info("Generated %d B-roll clips", len(generated_files))
        return generated_files

    def _generate_stock(
        self,
        query: str,
        output_dir: Path,
        filename_prefix: str,
        scene: Dict,
        prefer_video: bool = True,
    ) -> Optional[StockMediaResult]:
        """Generate B-roll using stock APIs."""
        logger.info("Downloading stock media: %s", query)
        try:
            from VideoForge.config.config_manager import Config

            quality_check_enabled = Config.get("enable_quality_check")
            if quality_check_enabled is None:
                quality_check_enabled = True
            else:
                quality_check_enabled = bool(quality_check_enabled)
            try:
                min_quality_score = float(Config.get("min_quality_score") or 0.6)
            except (TypeError, ValueError):
                min_quality_score = 0.6
        except Exception:
            quality_check_enabled = False
            min_quality_score = 0.6

        logger.info(
            "Quality check enabled=%s (min=%.2f)",
            "on" if quality_check_enabled else "off",
            min_quality_score,
        )

        def _download_candidate(
            candidate: Dict[str, object], name_suffix: str
        ) -> Optional[StockMediaResult]:
            media_type = str(candidate.get("media_type") or "")
            ext = ".mp4" if media_type == "video" else ".jpg"
            save_path = output_dir / f"{name_suffix}{ext}"
            if not self.stock.download_media(str(candidate.get("url") or ""), save_path):
                return None
            metadata = self.stock._build_metadata(candidate, query)
            return StockMediaResult(
                path=save_path,
                metadata=metadata,
                media_type=media_type or ("video" if ext == ".mp4" else "image"),
            )

        candidates = self.stock.search_candidates(
            query, prefer_video=prefer_video, per_page=3
        )
        if not candidates:
            return None

        candidate_media = str(candidates[0].get("media_type") or "")
        if not quality_check_enabled or candidate_media != "video":
            return _download_candidate(candidates[0], filename_prefix)

        best: Optional[StockMediaResult] = None
        best_score = -1.0
        downloaded_paths: List[Path] = []

        for idx, candidate in enumerate(candidates, start=1):
            result = _download_candidate(candidate, f"{filename_prefix}_c{idx}")
            if not result:
                continue
            downloaded_paths.append(result.path)
            if result.media_type != "video":
                continue
            try:
                from VideoForge.broll.quality_checker import VideoQualityChecker

                checker = VideoQualityChecker()
                quality = checker.check_video_quality(result.path)
                score = float(quality.get("quality_score", 0.0))
            except Exception as exc:
                logger.warning("Quality check failed: %s", exc)
                return result

            logger.info(
                "Quality check score=%.2f (min=%.2f) for %s",
                score,
                min_quality_score,
                result.path.name,
            )

            if score >= min_quality_score and score > best_score:
                best_score = score
                best = result

        if not best:
            for path in downloaded_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            logger.error(
                "Failed to find quality stock video after %d candidates",
                len(downloaded_paths),
            )
            return None

        final_path = output_dir / f"{filename_prefix}{best.path.suffix}"
        if best.path != final_path:
            try:
                os.replace(best.path, final_path)
                best.path = final_path
            except Exception:
                pass

        for path in downloaded_paths:
            if best.path == path:
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

        logger.info("Stock video quality: %.2f", best_score)
        return best

    def _generate_ai(
        self,
        scene_description: str,
        output_dir: Path,
        filename_prefix: str,
        scene: Dict,
    ) -> Optional[Path]:
        """Generate B-roll using ComfyUI."""
        if not self.comfyui.ensure_running():
            logger.warning("ComfyUI not available")
            return None

        # Use LLM-generated prompts if available, fallback to template
        positive = scene.get("comfyui_positive")
        negative = scene.get("comfyui_negative")
        seed = scene.get("comfyui_seed")
        width = scene.get("comfyui_width")
        height = scene.get("comfyui_height")

        if not width or not height:
            try:
                from VideoForge.config.config_manager import Config

                res = str(Config.get("comfyui_resolution") or "1920x1024")
            except Exception:
                res = "1920x1024"
            try:
                w_str, h_str = res.lower().split("x", 1)
                width = int(w_str.strip())
                height = int(h_str.strip())
            except Exception:
                width, height = 1920, 1024

        if not positive:
            # Fallback: build prompts from scene analysis metadata
            camera = scene.get("camera", "medium shot")
            mood = scene.get("mood", "professional")
            lighting = scene.get("lighting", "natural")

            positive = (
                f"{scene_description}, {camera}, {mood} mood, {lighting} lighting, "
                "professional photography, 8k resolution, highly detailed, "
                "cinematic, photorealistic"
            )

        if not negative:
            negative = (
                "cartoon, illustration, anime, low quality, blurry, amateur, "
                "artificial, oversaturated, cluttered, lowres, jpeg artifacts, "
                "text, watermark"
            )

        if seed is None:
            seed = self.comfyui.random_seed()
            scene["comfyui_seed"] = seed

        scene["comfyui_positive"] = positive
        scene["comfyui_negative"] = negative

        logger.info("Generating AI image: %s", filename_prefix)

        # Generate image
        success = self.comfyui.generate_and_wait(
            positive,
            negative,
            filename_prefix,
            seed=seed,
            width=int(width),
            height=int(height),
            timeout_sec=300,
        )

        if not success:
            logger.error("ComfyUI generation failed: %s", filename_prefix)
            return None

        # Find output images in ComfyUI output folder
        output_files = self.comfyui.find_output_images(
            filename_prefix, timeout_sec=60.0
        )

        if not output_files:
            logger.warning("ComfyUI output not found: %s", filename_prefix)
            return None

        comfyui_output = output_files[0]

        # Copy to project output directory
        project_output = output_dir / f"{filename_prefix}.png"
        try:
            shutil.copy2(comfyui_output, project_output)
            logger.info("Copied ComfyUI output: %s -> %s", comfyui_output.name, project_output.name)
            return project_output
        except Exception as exc:
            logger.error("Failed to copy ComfyUI output: %s", exc)
            return None

    def _image_to_video(
        self, image_path: Path, duration: float = 5.0, output_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Convert static image to video using FFmpeg.

        Args:
            image_path: Input image path
            duration: Video duration in seconds
            output_name: Output filename (default: same as image with .mp4)

        Returns:
            Path to output video, or None if failed
        """
        if not output_name:
            output_name = image_path.stem + ".mp4"

        output_path = image_path.parent / output_name

        # FFmpeg command: loop image for duration
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-c:v",
            "libx264",
            "-t",
            str(duration),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            str(output_path),
        ]

        try:
            logger.info("Converting image to video: %s", output_name)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and output_path.exists():
                logger.info("Video created: %s", output_path.name)
                return output_path

            logger.error("FFmpeg failed: %s", result.stderr)
            return None

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout")
            return None
        except Exception as exc:
            logger.error("FFmpeg error: %s", exc)
            return None

    @staticmethod
    def _wait_for_file_ready(
        file_path: Path, timeout_sec: float = 5.0, interval_sec: float = 0.2
    ) -> bool:
        """Wait until a file exists and size is stable."""
        start = time.time()
        last_size = -1
        stable = 0
        while (time.time() - start) < timeout_sec:
            if file_path.exists():
                try:
                    size = file_path.stat().st_size
                except OSError:
                    size = -1
                if size > 0 and size == last_size:
                    stable += 1
                    if stable >= 3:
                        return True
                else:
                    stable = 0
                    last_size = size
            time.sleep(interval_sec)
        return file_path.exists() and file_path.stat().st_size > 0

    def _embed_metadata(self, video_path: Path, metadata: Dict[str, object]) -> Path:
        """
        Embed metadata as JSON in MP4 comment field.

        Args:
            video_path: Video file path
            metadata: Metadata dict to embed
        """
        temp_path = video_path.with_suffix(".tmp.mp4")
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-metadata",
            f"comment={metadata_json}",
            str(temp_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and temp_path.exists():
                final_path = self._replace_with_retry(video_path, temp_path)
                if final_path != video_path:
                    logger.warning(
                        "Metadata embed fallback path used: %s", final_path.name
                    )
                else:
                    logger.debug("Metadata embedded in %s", video_path.name)
                return final_path
            logger.warning("Metadata embed failed: %s", result.stderr)
            if temp_path.exists():
                temp_path.unlink()
            return video_path

        except Exception as exc:
            logger.error("Metadata embed error: %s", exc)
            if temp_path.exists():
                temp_path.unlink()
            return video_path

    def _replace_with_retry(self, target_path: Path, temp_path: Path) -> Path:
        for _ in range(3):
            try:
                os.replace(temp_path, target_path)
                return target_path
            except PermissionError:
                time.sleep(0.2)
            except Exception:
                break
        fallback_path = target_path.with_name(target_path.stem + "_meta.mp4")
        try:
            shutil.copy2(temp_path, fallback_path)
            temp_path.unlink()
            return fallback_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            return target_path

    @staticmethod
    def _resolve_source_mode(source_mode: str, fallback_mode: str) -> Optional[str]:
        if source_mode == "library_only":
            return None
        if source_mode in {"library_stock", "stock_only"}:
            return "stock"
        if source_mode in {"library_ai", "ai_only"}:
            return "ai"
        if source_mode == "library_stock_ai":
            return "hybrid"
        return fallback_mode

    @staticmethod
    def _extract_search_tokens(text: str, limit: int = 3) -> List[str]:
        tokens = re.findall(r"[0-9A-Za-z가-힣]+", text)
        cleaned: List[str] = []
        for token in tokens:
            cleaned_token = token.strip().lower()
            if not cleaned_token or cleaned_token.isdigit():
                continue
            if len(cleaned_token) < 2:
                continue
            cleaned.append(cleaned_token)
        deduped: List[str] = []
        seen = set()
        for token in cleaned:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped[:limit]

    def extract_metadata(self, video_path: Path) -> Dict[str, object]:
        """
        Extract metadata from MP4 comment field via ffprobe.

        Args:
            video_path: Video file path

        Returns:
            Metadata dict (empty if not found)
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.warning("ffprobe failed for %s", video_path)
                return {}

            stdout = result.stdout or ""
            if not stdout.strip():
                return {}
            data = json.loads(stdout)
            comment = data.get("format", {}).get("tags", {}).get("comment")

            if comment in (None, ""):
                return {}

            if isinstance(comment, (dict, list)):
                return coerce_metadata_dict(comment)
            if isinstance(comment, (bytes, bytearray)):
                comment = comment.decode("utf-8", errors="ignore")
            return coerce_metadata_dict(json.loads(str(comment)))

        except json.JSONDecodeError:
            logger.warning("JSON parse failed for %s", video_path)
            return {}
        except Exception as exc:
            logger.error("Metadata extraction error: %s", exc)
            return {}
