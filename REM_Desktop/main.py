
import json
import sys
import uuid

import requests

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 120

APP_STYLE = """
QMainWindow, QWidget {
    background: #0b1020;
    color: #e5e7eb;
    font-family: "Segoe UI";
    font-size: 13px;
}
QFrame#sidebar {
    background: #11182a;
    border: 1px solid #26334d;
    border-radius: 12px;
}
QFrame#panel {
    background: #11182a;
    border: 1px solid #26334d;
    border-radius: 10px;
}
QLabel#muted {
    color: #9aa8c2;
}
QLabel#heading {
    font-size: 25px;
    font-weight: 700;
}
QLabel#metric {
    font-size: 22px;
    font-weight: 700;
    color: #7dd3fc;
}
QLabel#status {
    background: #11182a;
    border: 1px solid #26334d;
    border-radius: 8px;
    padding: 12px;
}
QPushButton {
    background: #1c2941;
    border: 1px solid #344562;
    border-radius: 7px;
    padding: 10px;
    text-align: left;
}
QPushButton:hover {
    background: #293b5b;
}
QPushButton#primary {
    background: #2563eb;
    border: none;
    font-weight: 600;
    text-align: center;
}
QPushButton#primary:hover {
    background: #1d4ed8;
}
QPlainTextEdit {
    background: #080d19;
    border: 1px solid #344562;
    border-radius: 8px;
    padding: 10px;
    selection-background-color: #2563eb;
}
"""


class REMDesktop(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("REM | Cyber Threat Intelligence")
        self.resize(1250, 820)
        self.setMinimumSize(1000, 680)

        self.setStyleSheet(APP_STYLE)

        self.last_response = None
        self.last_health = None
        self.busy = False

        self.pages = QStackedWidget()
        self.dashboard = self.build_dashboard()
        self.analysis = self.build_analysis()

        self.pages.addWidget(self.dashboard)
        self.pages.addWidget(self.analysis)

        self.build_shell()
        self.refresh_health()

        # Refresh the API status every 10 seconds.
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.refresh_health)
        self.health_timer.start(10000)

    def build_shell(self):
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(225)

        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(10)

        brand = QLabel("REM")
        brand.setFont(QFont("Segoe UI", 25, QFont.Weight.Bold))
        side.addWidget(brand)

        tagline = QLabel("Cyber Defense Platform")
        tagline.setObjectName("muted")
        side.addWidget(tagline)

        dashboard_button = QPushButton("Dashboard")
        dashboard_button.clicked.connect(
            lambda: self.pages.setCurrentIndex(0)
        )
        side.addWidget(dashboard_button)

        analysis_button = QPushButton("Threat Analysis")
        analysis_button.clicked.connect(
            lambda: self.pages.setCurrentIndex(1)
        )
        side.addWidget(analysis_button)

        history_button = QPushButton("Latest Result")
        history_button.clicked.connect(self.show_latest_result)
        side.addWidget(history_button)

        side.addStretch()

        side.addWidget(QLabel("LOCAL INFERENCE"))

        self.sidebar_api = QLabel("API: Checking...")
        self.sidebar_api.setObjectName("muted")
        side.addWidget(self.sidebar_api)

        side.addWidget(QLabel("Retraining: Disabled"))
        side.addWidget(QLabel("Autonomous response: Disabled"))

        outer.addWidget(sidebar)
        outer.addWidget(self.pages, 1)

        self.setCentralWidget(root)

    @staticmethod
    def create_panel(title, value, detail):
        frame = QFrame()
        frame.setObjectName("panel")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_label = QLabel(title.upper())
        title_label.setObjectName("muted")

        value_label = QLabel(value)
        value_label.setObjectName("metric")
        value_label.setWordWrap(True)

        detail_label = QLabel(detail)
        detail_label.setObjectName("muted")
        detail_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        layout.addStretch()

        return frame, value_label, detail_label

    @staticmethod
    def add_heading(layout, title, subtitle):
        heading = QLabel(title)
        heading.setObjectName("heading")
        layout.addWidget(heading)

        description = QLabel(subtitle)
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)

    def build_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        self.add_heading(
            layout,
            "Security Operations Dashboard",
            "REM hybrid cyber threat intelligence and inference.",
        )

        grid = QGridLayout()
        grid.setSpacing(12)

        panel, self.dashboard_api, _ = self.create_panel(
            "API Status", "CHECKING", "Local FastAPI service"
        )
        grid.addWidget(panel, 0, 0)

        panel, _, _ = self.create_panel(
            "Model Stack",
            "4 Models",
            "Random Forest, LSTM, Isolation Forest and Hybrid",
        )
        grid.addWidget(panel, 0, 1)

        panel, _, _ = self.create_panel(
            "Self-Learning",
            "Disabled",
            "Verified human feedback is required",
        )
        grid.addWidget(panel, 1, 0)

        panel, _, _ = self.create_panel(
            "Autonomous Response",
            "Disabled",
            "No automatic remediation is enabled",
        )
        grid.addWidget(panel, 1, 1)

        layout.addLayout(grid)

        status_heading = QLabel("System Status")
        layout.addWidget(status_heading)

        self.health_details = QLabel("Checking backend...")
        self.health_details.setObjectName("status")
        self.health_details.setWordWrap(True)
        layout.addWidget(self.health_details)

        refresh_button = QPushButton("Refresh API Status")
        refresh_button.clicked.connect(self.refresh_health)
        layout.addWidget(refresh_button)

        open_analysis = QPushButton("Open Threat Analysis")
        open_analysis.setObjectName("primary")
        open_analysis.clicked.connect(
            lambda: self.pages.setCurrentIndex(1)
        )
        layout.addWidget(open_analysis)

        layout.addStretch()
        return page

    def build_analysis(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        self.add_heading(
            layout,
            "Threat Analysis",
            "Submit an ordered API-call sequence to the existing REM API.",
        )

        layout.addWidget(QLabel("Sample ID"))

        self.sample_id = QPlainTextEdit()
        self.sample_id.setMaximumHeight(45)
        self.sample_id.setPlaceholderText("Unique sample ID")
        self.sample_id.setPlainText(
            "REM-DESKTOP-" + uuid.uuid4().hex[:10].upper()
        )
        layout.addWidget(self.sample_id)

        layout.addWidget(
            QLabel("API calls — enter one call per line")
        )

        self.api_calls = QPlainTextEdit()
        self.api_calls.setMinimumHeight(105)
        self.api_calls.setPlaceholderText(
            "GetSystemInfo\nGetShortPathNameW\nCoUninitialize"
        )
        self.api_calls.setPlainText(
            "GetSystemInfo\nGetShortPathNameW\nCoUninitialize"
        )
        layout.addWidget(self.api_calls)

        button_row = QHBoxLayout()

        self.run_button = QPushButton("Run REM Analysis")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_analysis)
        button_row.addWidget(self.run_button)

        new_id_button = QPushButton("Generate New Sample ID")
        new_id_button.clicked.connect(self.generate_sample_id)
        button_row.addWidget(new_id_button)

        layout.addLayout(button_row)

        summary_title = QLabel("Prediction Summary")
        layout.addWidget(summary_title)

        self.result_summary = QLabel(
            "No analysis has been run yet."
        )
        self.result_summary.setObjectName("status")
        self.result_summary.setWordWrap(True)
        layout.addWidget(self.result_summary)

        score_heading = QLabel("Model Scores")
        layout.addWidget(score_heading)

        self.score_labels = {}
        self.score_details = {}

        score_grid = QGridLayout()
        score_grid.setSpacing(10)

        definitions = [
            (
                "random_forest",
                "Random Forest",
                "Malicious-class probability",
            ),
            (
                "lstm",
                "LSTM",
                "Model output probability",
            ),
            (
                "isolation_forest",
                "Isolation Forest",
                "Anomaly score — not a probability",
            ),
            (
                "hybrid",
                "Hybrid REM",
                "Experimental combined score",
            ),
        ]

        for index, (key, title, description) in enumerate(definitions):
            frame, value_label, detail_label = self.create_panel(
                title, "—", description
            )
            self.score_labels[key] = value_label
            self.score_details[key] = detail_label
            score_grid.addWidget(frame, index // 2, index % 2)

        layout.addLayout(score_grid)

        diagnostic_heading = QLabel("Review and Diagnostic Details")
        layout.addWidget(diagnostic_heading)

        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlaceholderText(
            "The API response and review status will appear here."
        )
        self.results.setMinimumHeight(100)
        layout.addWidget(self.results, 1)

        return page

    def generate_sample_id(self):
        self.sample_id.setPlainText(
            "REM-DESKTOP-" + uuid.uuid4().hex[:10].upper()
        )

    def refresh_health(self):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=3,
            )
            response.raise_for_status()
            data = response.json()
            self.last_health = data

            status = str(data.get("status", "unknown")).upper()
            self.dashboard_api.setText(status)
            self.sidebar_api.setText(f"API: {status}")

            self.health_details.setText(
                f"API status: {data.get('status', 'unknown')}\n"
                f"Service: {data.get('service', 'REM API')}\n"
                f"Training enabled: "
                f"{data.get('training_enabled', 'unknown')}\n"
                f"Autonomous response enabled: "
                f"{data.get('autonomous_response_enabled', 'unknown')}"
            )

        except requests.RequestException as exc:
            self.last_health = None
            self.dashboard_api.setText("OFFLINE")
            self.sidebar_api.setText("API: OFFLINE")
            self.health_details.setText(
                "REM API is unavailable.\n"
                "Start the backend and select Refresh API Status.\n"
                f"Error: {type(exc).__name__}"
            )

    def run_analysis(self):
        if self.busy:
            return

        sample_id = self.sample_id.toPlainText().strip()
        api_calls = [
            line.strip()
            for line in self.api_calls.toPlainText().splitlines()
            if line.strip()
        ]

        if not sample_id:
            QMessageBox.warning(
                self,
                "Missing sample ID",
                "Enter a unique sample ID.",
            )
            return

        if not api_calls:
            QMessageBox.warning(
                self,
                "Missing API calls",
                "Enter at least one API call.",
            )
            return

        payload = {
            "sample_id": sample_id,
            "api_calls": api_calls,
        }

        self.busy = True
        self.run_button.setEnabled(False)
        self.run_button.setText("Running inference...")
        self.result_summary.setText("Inference is in progress...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            self.last_response = data
            self.render_prediction(data)

        except requests.HTTPError as exc:
            response = exc.response
            message = f"HTTP {response.status_code}"

            try:
                body = response.json()
                message += "\n" + json.dumps(body, indent=2)
            except ValueError:
                message += "\n" + response.text[:3000]

            self.result_summary.setText(
                "Inference failed. See diagnostic details below."
            )
            self.results.setPlainText(message)

        except requests.RequestException as exc:
            self.result_summary.setText("Could not contact the REM API.")
            self.results.setPlainText(
                f"Request error: {type(exc).__name__}\n{exc}"
            )

        except (ValueError, TypeError, KeyError) as exc:
            self.result_summary.setText(
                "The API responded, but its result could not be parsed."
            )
            self.results.setPlainText(
                f"Response parsing error: {type(exc).__name__}\n{exc}"
            )

        finally:
            QApplication.restoreOverrideCursor()
            self.busy = False
            self.run_button.setEnabled(True)
            self.run_button.setText("Run REM Analysis")

    def render_prediction(self, data):
        # Current bridge response shape:
        # {"inference": {...}, "queued_record": {...}, ...}
        inference = data.get("inference", {})
        models = inference.get("models", {})
        queued = data.get("queued_record", {})

        model_keys = {
            "random_forest": "random_forest",
            "lstm": "lstm",
            "isolation_forest": "isolation_forest",
            "hybrid": "hybrid",
        }

        for ui_key, response_key in model_keys.items():
            result = models.get(response_key, {})
            score = result.get("score")
            threshold = result.get("threshold")
            decision = result.get("decision", "unknown")
            score_type = result.get("score_type", "unspecified")

            if score is None:
                self.score_labels[ui_key].setText("N/A")
                self.score_details[ui_key].setText(
                    "Score not present in API response"
                )
                continue

            try:
                numeric_score = float(score)
            except (ValueError, TypeError):
                self.score_labels[ui_key].setText("N/A")
                self.score_details[ui_key].setText(
                    f"Invalid score value: {score}"
                )
                continue

            if ui_key == "isolation_forest":
                display_score = f"{numeric_score:.4f}"
            else:
                display_score = f"{numeric_score * 100:.1f}%"

            self.score_labels[ui_key].setText(display_score)
            self.score_details[ui_key].setText(
                f"Decision: {decision}\n"
                f"Threshold: {threshold}\n"
                f"Type: {score_type}"
            )

        hybrid = models.get("hybrid", {})
        hybrid_decision = hybrid.get("decision", "unknown")
        review_status = queued.get("review_status", "not returned")
        verified_label = queued.get("verified_label")
        eligible = queued.get("eligible_for_training", False)

        self.result_summary.setText(
            f"Hybrid decision: {str(hybrid_decision).upper()}\n"
            "Model status: EXPERIMENTAL\n"
            f"Review status: {review_status}\n"
            f"Verified label: {verified_label}\n"
            f"Eligible for training: {eligible}\n"
            "No automated retraining or response is initiated by this UI."
        )

        self.results.setPlainText(json.dumps(data, indent=2))

    def show_latest_result(self):
        if self.last_response is None:
            QMessageBox.information(
                self,
                "No result available",
                "Run a threat analysis first.",
            )
            return

        self.pages.setCurrentIndex(1)
        self.results.setPlainText(
            json.dumps(self.last_response, indent=2)
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("REM Cyber Threat Intelligence")
    window = REMDesktop()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
