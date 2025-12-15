"""Enhanced feature labeling component with categories."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html


class NeuronCategory(Enum):
    """Categories for neuron classification."""

    NONE = "none"
    MECHANICAL = "mechanical"
    TACTICAL = "tactical"
    STRATEGIC = "strategic"


@dataclass
class FeatureLabel:
    """Complete label information for a feature."""

    name: str = ""
    category: str = NeuronCategory.NONE.value
    notes: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureLabel":
        """Create from dictionary."""
        return cls(**data)


class FeatureLabelsManager:
    """Enhanced feature labeling with categories and metadata."""

    def __init__(
        self, storage_path: Optional[Path] = None, model_type: str = "full"
    ):
        """Initialize the feature labels manager.

        Args:
            storage_path: Path to store feature labels JSON.
                         Defaults to dashboard directory with model-specific file.
            model_type: Type of model ("full" or "ultra")
        """
        self.model_type = model_type
        if storage_path is None:
            # Use model-specific label file
            filename = f"feature_labels_{model_type}.json"
            storage_path = Path(__file__).parent.parent / filename
        self.storage_path = storage_path
        self.feature_labels = self._load_labels()

        # Keep compatibility with old feature_names.json
        self._migrate_from_names_if_needed()

    def _load_labels(self) -> Dict[int, FeatureLabel]:
        """Load feature labels from storage."""
        if not self.storage_path.exists():
            return {}
        with open(self.storage_path, "r") as f:
            data = json.load(f)
        return {int(k): FeatureLabel.from_dict(v) for k, v in data.items()}

    def _migrate_from_names_if_needed(self):
        """Migrate from old feature_names.json if it exists."""
        old_path = self.storage_path.parent / "feature_names.json"
        if old_path.exists() and not self.feature_labels:
            with open(old_path, "r") as f:
                old_data = json.load(f)
            for k, name in old_data.items():
                self.feature_labels[int(k)] = FeatureLabel(
                    name=name, timestamp=datetime.now().isoformat()
                )
            self.save_labels()

    def save_labels(self) -> None:
        """Save feature labels to storage."""
        data = {str(k): v.to_dict() for k, v in self.feature_labels.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_label(self, feature_id: int) -> Optional[FeatureLabel]:
        """Get the label for a feature ID."""
        return self.feature_labels.get(feature_id)

    def set_label(self, feature_id: int, label: FeatureLabel) -> None:
        """Set the label for a feature ID."""
        label.timestamp = datetime.now().isoformat()
        self.feature_labels[feature_id] = label
        self.save_labels()

    def update_label(self, feature_id: int, **kwargs) -> None:
        """Update specific fields of a label."""
        label = self.get_label(feature_id) or FeatureLabel()
        for key, value in kwargs.items():
            if hasattr(label, key):
                setattr(label, key, value)
        self.set_label(feature_id, label)

    def get_display_name(self, feature_id: int) -> str:
        """Get display name with category if available."""
        label = self.get_label(feature_id)
        if label and label.name:
            category_str = ""
            if label.category and label.category != NeuronCategory.NONE.value:
                category_str = f" [{label.category.upper()}]"
            return f"Feature {feature_id}: {label.name}{category_str}"
        return f"Feature {feature_id}"

    def get_by_category(
        self, category: NeuronCategory
    ) -> Dict[int, FeatureLabel]:
        """Get all features with a specific category."""
        return {
            fid: label
            for fid, label in self.feature_labels.items()
            if label.category == category.value
        }

    def search_labels(self, query: str) -> Dict[int, str]:
        """Search feature labels by query (searches ID, name, category, notes)."""
        query = query.lower().strip()
        results = {}

        # Search by ID
        try:
            feature_id = int(query)
            if 0 <= feature_id < 10000:  # Assuming reasonable feature count
                results[feature_id] = self.get_display_name(feature_id)
        except ValueError:
            pass

        # Search by content
        for fid, label in self.feature_labels.items():
            if (
                query in label.name.lower()
                or query in label.category.lower()
                or query in label.notes.lower()
            ):
                results[fid] = self.get_display_name(fid)

        return results

    def get_statistics(self) -> Dict[str, int]:
        """Get labeling statistics."""
        stats = {
            "total_labeled": len(self.feature_labels),
            "with_names": sum(
                1 for l in self.feature_labels.values() if l.name
            ),
            "mechanical": len(self.get_by_category(NeuronCategory.MECHANICAL)),
            "tactical": len(self.get_by_category(NeuronCategory.TACTICAL)),
            "strategic": len(self.get_by_category(NeuronCategory.STRATEGIC)),
            "uncategorized": sum(
                1
                for l in self.feature_labels.values()
                if l.category == NeuronCategory.NONE.value
            ),
        }
        return stats


def create_feature_label_editor(
    feature_id: int, label_manager: FeatureLabelsManager
) -> dbc.Card:
    """Create the feature label editor component."""
    label = label_manager.get_label(feature_id) or FeatureLabel()

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Feature Labels", className="mb-0")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Feature Name"),
                                    dbc.Input(
                                        id="feature-name-input",
                                        type="text",
                                        value=label.name,
                                        placeholder="Enter descriptive name...",
                                        debounce=True,
                                    ),
                                ],
                                md=12,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Category"),
                                    dbc.RadioItems(
                                        id="feature-category-radio",
                                        options=[
                                            {
                                                "label": "None",
                                                "value": NeuronCategory.NONE.value,
                                            },
                                            {
                                                "label": "Mechanical",
                                                "value": NeuronCategory.MECHANICAL.value,
                                            },
                                            {
                                                "label": "Tactical",
                                                "value": NeuronCategory.TACTICAL.value,
                                            },
                                            {
                                                "label": "Strategic",
                                                "value": NeuronCategory.STRATEGIC.value,
                                            },
                                        ],
                                        value=label.category,
                                        inline=True,
                                    ),
                                ],
                                md=12,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Notes"),
                                    dbc.Textarea(
                                        id="feature-notes-textarea",
                                        value=label.notes,
                                        placeholder="Additional notes about this feature...",
                                        style={"height": "100px"},
                                        debounce=True,
                                    ),
                                ],
                                md=12,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Save Labels",
                                        id="save-feature-labels-btn",
                                        color="primary",
                                        className="w-100",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Clear Labels",
                                        id="clear-feature-labels-btn",
                                        color="danger",
                                        outline=True,
                                        className="w-100",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    html.Div(id="feature-labels-feedback", className="mt-2"),
                    # Hidden store to trigger updates
                    dcc.Store(id="feature-labels-updated"),
                ]
            ),
        ],
        className="mb-3",
    )


def create_labeling_statistics(label_manager: FeatureLabelsManager) -> dbc.Card:
    """Create a statistics display for labeling progress."""
    stats = label_manager.get_statistics()

    # Calculate percentages for progress bars
    total = stats["total_labeled"] or 1  # Avoid division by zero
    mechanical_pct = (stats["mechanical"] / total) * 100 if total > 0 else 0
    tactical_pct = (stats["tactical"] / total) * 100 if total > 0 else 0
    strategic_pct = (stats["strategic"] / total) * 100 if total > 0 else 0
    uncategorized_pct = (
        (stats["uncategorized"] / total) * 100 if total > 0 else 0
    )

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Labeling Statistics", className="mb-0")),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.P(
                                [
                                    html.Strong("Total Labeled: "),
                                    f"{stats['total_labeled']} features",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                [
                                    html.Strong("With Names: "),
                                    f"{stats['with_names']} features",
                                ],
                                className="mb-1",
                            ),
                            html.Hr(),
                            html.P(
                                html.Strong("Categories:"), className="mb-2"
                            ),
                            html.Div(
                                [
                                    dbc.Progress(
                                        value=mechanical_pct,
                                        color="info",
                                        label=f"Mechanical: {stats['mechanical']}",
                                        className="mb-1",
                                        style={"height": "20px"},
                                    ),
                                    dbc.Progress(
                                        value=tactical_pct,
                                        color="warning",
                                        label=f"Tactical: {stats['tactical']}",
                                        className="mb-1",
                                        style={"height": "20px"},
                                    ),
                                    dbc.Progress(
                                        value=strategic_pct,
                                        color="success",
                                        label=f"Strategic: {stats['strategic']}",
                                        className="mb-1",
                                        style={"height": "20px"},
                                    ),
                                    dbc.Progress(
                                        value=uncategorized_pct,
                                        color="secondary",
                                        label=f"Uncategorized: {stats['uncategorized']}",
                                        className="mb-1",
                                        style={"height": "20px"},
                                    ),
                                ],
                                className="mb-2",
                            ),
                        ]
                    )
                ]
            ),
        ],
        className="mb-3",
    )
