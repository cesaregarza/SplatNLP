"""Client for the activation server.

Provides transparent fallback to direct database access if server is unavailable.
"""

import logging
import os
from typing import Any

import httpx
import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://127.0.0.1:8765"


class ServerBackedDatabase:
    """Database wrapper that uses the activation server when available.

    Provides the same interface as EfficientFSDatabase but fetches data
    from the activation server for faster access when it's running.
    Falls back to direct database access if server is unavailable.
    """

    def __init__(
        self,
        data_dir: str,
        examples_dir: str,
        server_url: str | None = None,
    ):
        self.data_dir = data_dir
        self.examples_dir = examples_dir
        self.server_url = server_url or os.environ.get(
            "ACTIVATION_SERVER_URL", DEFAULT_SERVER_URL
        )
        self._server_available: bool | None = None
        self._fallback_db = None
        self._feature_ids_cache: list[int] | None = None

    def _check_server(self) -> bool:
        """Check if the activation server is available."""
        if self._server_available is not None:
            return self._server_available

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.server_url}/health")
                if resp.status_code == 200:
                    logger.info(f"Using activation server at {self.server_url}")
                    self._server_available = True
                    return True
        except Exception:
            pass

        logger.info("Activation server not available, using direct DB access")
        self._server_available = False
        return False

    def _get_fallback_db(self):
        """Lazy load the fallback database."""
        if self._fallback_db is None:
            from splatnlp.dashboard.efficient_fs_database import (
                EfficientFSDatabase,
            )

            self._fallback_db = EfficientFSDatabase(
                data_dir=self.data_dir,
                examples_dir=self.examples_dir,
            )
        return self._fallback_db

    def get_all_feature_ids(self) -> list[int]:
        """Get all available feature IDs."""
        if self._feature_ids_cache is not None:
            return self._feature_ids_cache

        if self._check_server():
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(f"{self.server_url}/features")
                    resp.raise_for_status()
                    self._feature_ids_cache = resp.json()["feature_ids"]
                    return self._feature_ids_cache
            except Exception as e:
                logger.warning(f"Server feature list failed: {e}")
                self._server_available = False

        self._feature_ids_cache = self._get_fallback_db().get_all_feature_ids()
        return self._feature_ids_cache

    def get_all_feature_activations_for_pagerank(
        self,
        feature_id: int,
        *,
        limit: int | None = None,
        sample_frac: float | None = None,
        include_abilities: bool = True,
    ) -> pl.DataFrame:
        """Get activation data for a feature as a Polars DataFrame.

        This is the main method used by experiment runners.
        """
        if self._check_server():
            try:
                # Use longer timeout for large datasets (100k+ rows can be 10+ MB JSON)
                with httpx.Client(timeout=120.0) as client:
                    params = {}
                    if limit is not None:
                        params["limit"] = limit
                    if sample_frac is not None:
                        params["sample_frac"] = sample_frac
                    if include_abilities is False:
                        params["include_abilities"] = "false"

                    resp = client.get(
                        f"{self.server_url}/activations/{feature_id}",
                        params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    logger.debug(
                        f"Fetched {data['n_rows']} rows from server "
                        f"in {data['load_time_ms']:.1f}ms"
                    )
                    # Convert back to DataFrame
                    return pl.DataFrame(data["data"])
            except Exception as e:
                logger.warning(f"Server fetch failed, falling back: {e}")
                self._server_available = False

        return self._get_fallback_db().get_all_feature_activations_for_pagerank(
            feature_id,
            include_negative=True,
            limit=limit,
            sample_frac=sample_frac,
            include_abilities=include_abilities,
        )

    # Delegate other methods to the fallback database
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown methods to the fallback database."""
        return getattr(self._get_fallback_db(), name)


class ActivationClient:
    """Client for fetching activation data from the server.

    Falls back to direct database access if server is unavailable.
    """

    def __init__(
        self,
        server_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.server_url = server_url or os.environ.get(
            "ACTIVATION_SERVER_URL", DEFAULT_SERVER_URL
        )
        self.timeout = timeout
        self._server_available: bool | None = None
        self._fallback_db = None

    def _check_server(self) -> bool:
        """Check if the activation server is available."""
        if self._server_available is not None:
            return self._server_available

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.server_url}/health")
                if resp.status_code == 200:
                    logger.info(
                        f"Activation server available at {self.server_url}"
                    )
                    self._server_available = True
                    return True
        except Exception:
            pass

        logger.info("Activation server not available, using direct DB access")
        self._server_available = False
        return False

    def _get_fallback_db(self):
        """Lazy load the fallback database."""
        if self._fallback_db is None:
            from splatnlp.dashboard.efficient_fs_database import (
                EfficientFSDatabase,
            )
            from splatnlp.mechinterp.skill_helpers.context_loader import (
                ULTRA_MODEL_PATHS,
            )

            self._fallback_db = EfficientFSDatabase(
                data_dir=ULTRA_MODEL_PATHS["data_dir"],
                examples_dir=ULTRA_MODEL_PATHS["examples_dir"],
            )
        return self._fallback_db

    def get_activations(
        self, feature_id: int, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get activation data for a feature.

        Args:
            feature_id: The SAE feature ID
            limit: Optional limit on rows

        Returns:
            List of activation dicts with keys: activation, ability_input_tokens, weapon_id
        """
        if self._check_server():
            return self._fetch_from_server(feature_id, limit)
        else:
            return self._fetch_from_db(feature_id, limit)

    def _fetch_from_server(
        self, feature_id: int, limit: int | None
    ) -> list[dict[str, Any]]:
        """Fetch from the activation server."""
        try:
            url = f"{self.server_url}/activations/{feature_id}"
            if limit:
                url += f"?limit={limit}"

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                logger.debug(
                    f"Fetched {data['n_rows']} rows from server "
                    f"in {data['load_time_ms']:.1f}ms"
                )
                return data["data"]

        except Exception as e:
            logger.warning(f"Server fetch failed, falling back to DB: {e}")
            self._server_available = False
            return self._fetch_from_db(feature_id, limit)

    def _fetch_from_db(
        self, feature_id: int, limit: int | None
    ) -> list[dict[str, Any]]:
        """Fetch directly from the database."""
        db = self._get_fallback_db()
        df = db.get_all_feature_activations_for_pagerank(feature_id)
        if limit:
            df = df.head(limit)
        return df.to_dicts()

    def get_vocab(self) -> tuple[dict[str, int], dict[str, int]]:
        """Get vocabularies (vocab, weapon_vocab)."""
        if self._check_server():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(f"{self.server_url}/context/vocab")
                    resp.raise_for_status()
                    data = resp.json()
                    return data["vocab"], data["weapon_vocab"]
            except Exception as e:
                logger.warning(f"Server vocab fetch failed: {e}")

        # Fallback to loading from files
        from splatnlp.mechinterp.skill_helpers.context_loader import (
            load_context,
        )

        ctx = load_context("ultra")
        return ctx.vocab, ctx.weapon_vocab


# Singleton instance for convenience
_default_client: ActivationClient | None = None


def get_client() -> ActivationClient:
    """Get the default activation client."""
    global _default_client
    if _default_client is None:
        _default_client = ActivationClient()
    return _default_client


def get_activations(feature_id: int, limit: int | None = None) -> list[dict]:
    """Convenience function to get activations."""
    return get_client().get_activations(feature_id, limit)
