import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow


PROJECT_DIR = Path(__file__).resolve().parent


def load_settings() -> dict:
    settings_path = PROJECT_DIR / "config" / "settings.json"
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        QMessageBox.warning(
            None,
            "Settings unavailable",
            f"The settings file could not be loaded. Default values will be used.\n\n{error}",
        )
        return {}


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EyeControl AI")
    app.setStyle("Fusion")

    window = MainWindow(PROJECT_DIR, load_settings())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())