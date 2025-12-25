"""Stock media downloader for Pexels and Unsplash."""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from VideoForge.broll.metadata import BrollMetadata, now_iso

logger = logging.getLogger(__name__)


@dataclass
class StockMediaResult:
    path: Path
    metadata: BrollMetadata
    media_type: str


class StockMediaDownloader:
    """Download stock photos/videos from Pexels and Unsplash."""

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        unsplash_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize stock media downloader.

        Args:
            pexels_api_key: Pexels API key (from .env or Settings)
            unsplash_api_key: Unsplash API key (from .env or Settings)
        """
        self.pexels_key = pexels_api_key or os.getenv("PEXELS_API_KEY", "")
        self.unsplash_key = unsplash_api_key or os.getenv("UNSPLASH_API_KEY", "")

        self.pexels_video_url = "https://api.pexels.com/videos/search"
        self.pexels_photo_url = "https://api.pexels.com/v1/search"
        self.unsplash_url = "https://api.unsplash.com/search/photos"

    def search_pexels_videos(
        self, query: str, per_page: int = 5
    ) -> List[Dict[str, object]]:
        """
        Search Pexels for videos.

        Args:
            query: Search keywords
            per_page: Number of results

        Returns:
            List of video metadata dicts with 'url', 'width', 'height', 'duration'
        """
        if not self.pexels_key:
            logger.warning("Pexels API key not set")
            return []

        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",
        }
        headers = {"Authorization": self.pexels_key}

        try:
            response = requests.get(
                self.pexels_video_url, params=params, headers=headers, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                videos = data.get("videos", [])

                results = []
                for video in videos:
                    # Find HD or best quality video file
                    video_files = video.get("video_files", [])
                    hd_file = None

                    for vf in video_files:
                        if vf.get("quality") == "hd":
                            hd_file = vf
                            break

                    if not hd_file and video_files:
                        hd_file = video_files[0]

                    if hd_file:
                        results.append(
                            {
                                "url": hd_file.get("link", ""),
                                "width": hd_file.get("width", 1920),
                                "height": hd_file.get("height", 1080),
                                "duration": video.get("duration", 5.0),
                                "source": "pexels",
                                "media_type": "video",
                                "id": video.get("id", ""),
                                "page_url": video.get("url", ""),
                                "image": video.get("image", ""),
                                "photographer": (video.get("user") or {}).get("name", ""),
                            }
                        )

                logger.info("Pexels video search '%s': %d results", query, len(results))
                return results

            logger.warning("Pexels API error: %d %s", response.status_code, response.text)
            return []

        except Exception as exc:
            logger.error("Pexels video search failed: %s", exc)
            return []

    def search_pexels_photos(
        self, query: str, per_page: int = 5
    ) -> List[Dict[str, object]]:
        """
        Search Pexels for photos.

        Args:
            query: Search keywords
            per_page: Number of results

        Returns:
            List of photo metadata dicts with 'url', 'width', 'height'
        """
        if not self.pexels_key:
            logger.warning("Pexels API key not set")
            return []

        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",
        }
        headers = {"Authorization": self.pexels_key}

        try:
            response = requests.get(
                self.pexels_photo_url, params=params, headers=headers, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                photos = data.get("photos", [])

                results = []
                for photo in photos:
                    src = photo.get("src", {})
                    results.append(
                        {
                            "url": src.get("large2x", src.get("large", "")),
                            "width": photo.get("width", 1920),
                            "height": photo.get("height", 1080),
                            "source": "pexels",
                            "media_type": "image",
                            "id": photo.get("id", ""),
                            "description": photo.get("alt", ""),
                            "photographer": photo.get("photographer", ""),
                            "page_url": photo.get("url", ""),
                        }
                    )

                logger.info("Pexels photo search '%s': %d results", query, len(results))
                return results

            logger.warning("Pexels photo API error: %d %s", response.status_code, response.text)
            return []

        except Exception as exc:
            logger.error("Pexels photo search failed: %s", exc)
            return []

    def search_unsplash_photos(
        self, query: str, per_page: int = 5
    ) -> List[Dict[str, object]]:
        """
        Search Unsplash for photos.

        Args:
            query: Search keywords
            per_page: Number of results

        Returns:
            List of photo metadata dicts with 'url', 'width', 'height'
        """
        if not self.unsplash_key:
            logger.warning("Unsplash API key not set")
            return []

        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",
            "client_id": self.unsplash_key,
        }

        try:
            response = requests.get(self.unsplash_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                photos = data.get("results", [])

                results = []
                for photo in photos:
                    urls = photo.get("urls", {})
                    results.append(
                        {
                            "url": urls.get("full", urls.get("regular", "")),
                            "width": photo.get("width", 1920),
                            "height": photo.get("height", 1080),
                            "source": "unsplash",
                            "media_type": "image",
                            "id": photo.get("id", ""),
                            "description": photo.get("description") or photo.get("alt_description", ""),
                            "photographer": (photo.get("user") or {}).get("name", ""),
                            "tags": [t.get("title") for t in (photo.get("tags") or []) if t.get("title")],
                            "page_url": (photo.get("links") or {}).get("html", ""),
                        }
                    )

                logger.info("Unsplash search '%s': %d results", query, len(results))
                return results

            logger.warning("Unsplash API error: %d %s", response.status_code, response.text)
            return []

        except Exception as exc:
            logger.error("Unsplash search failed: %s", exc)
            return []

    def download_media(self, url: str, save_path: Path) -> bool:
        """
        Download media file from URL.

        Args:
            url: Media URL
            save_path: Local save path

        Returns:
            True if successful
        """
        try:
            # Add download tracking for Unsplash (required by API)
            if "unsplash" in url:
                url = url + "?ixlib=rb-4.0.3&dl="

            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                logger.info("Downloaded: %s", save_path.name)
                return True

            logger.warning("Download failed: HTTP %d", response.status_code)
            return False

        except Exception as exc:
            logger.error("Download error: %s", exc)
            return False

    def search_and_download(
        self,
        query: str,
        output_dir: Path,
        filename_prefix: str,
        prefer_video: bool = True,
    ) -> Optional[StockMediaResult]:
        """
        Search for media and download the best match.

        Args:
            query: Search keywords
            output_dir: Output directory
            filename_prefix: Filename prefix (e.g., "scene_001")
            prefer_video: Try videos first, fallback to photos

        Returns:
            Path to downloaded file, or None if failed
        """
        results = []

        # Try Pexels videos first if preferred
        if prefer_video and self.pexels_key:
            results = self.search_pexels_videos(query, per_page=3)
            if results:
                media = results[0]
                ext = ".mp4"
            else:
                # Fallback to photos
                prefer_video = False

        # Try photos if video not preferred or not found
        if not prefer_video:
            # Try Pexels photos
            if self.pexels_key:
                results = self.search_pexels_photos(query, per_page=3)

            # Fallback to Unsplash if Pexels fails
            if not results and self.unsplash_key:
                results = self.search_unsplash_photos(query, per_page=3)

            if results:
                media = results[0]
                ext = ".jpg"
            else:
                logger.warning("No media found for query: %s", query)
                return None

        # Download
        save_path = output_dir / f"{filename_prefix}{ext}"
        if self.download_media(media["url"], save_path):
            metadata = self._build_metadata(media, query)
            return StockMediaResult(
                path=save_path,
                metadata=metadata,
                media_type=str(media.get("media_type") or ("video" if ext == ".mp4" else "image")),
            )

        return None

    def _build_metadata(self, media: Dict[str, object], query: str) -> BrollMetadata:
        source = str(media.get("source") or "")
        media_type = str(media.get("media_type") or "")
        keywords = self._keywords_from_query(query)
        description = str(media.get("description") or query).strip()
        tags = media.get("tags") or []
        if tags:
            keywords.extend([str(t) for t in tags])
        keywords = self._unique_keywords(keywords)
        raw_metadata = {
            "provider_id": media.get("id"),
            "media_type": media_type,
            "page_url": media.get("page_url"),
            "photographer": media.get("photographer"),
            "query": query,
        }
        if media.get("image"):
            raw_metadata["image"] = media.get("image")
        duration = None
        try:
            if media.get("duration") is not None:
                duration = float(media.get("duration"))
        except (TypeError, ValueError):
            duration = None
        meta = BrollMetadata(
            source=source,
            keywords=keywords,
            description=description,
            visual_elements=list(keywords),
            created_at=now_iso(),
            duration=duration,
            raw_metadata=raw_metadata,
        )
        return meta

    @staticmethod
    def _keywords_from_query(query: str) -> List[str]:
        parts = [p.strip() for p in query.replace(",", " ").split()]
        return [p for p in parts if p]

    @staticmethod
    def _unique_keywords(items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            token = str(item).strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result
