"""Desktop entry point. CLI engines remain entirely independent of Qt."""

import argparse
import sys
import traceback
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from desktop.main_window import MainWindow


def main(argv=None):
    parser = argparse.ArgumentParser(description="Arknights Zero desktop control center")
    parser.add_argument("--quit-after", type=float, help="Close safely after N seconds (smoke checks)")
    parser.add_argument("--config", help="Open a specific training configuration without starting it")
    parser.add_argument("--review-account", action="store_true", help="Open the existing account review entry")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Arknights Zero")
    app.setOrganizationName("Arknights Zero Research")
    try:
        window = MainWindow()
    except Exception:
        text = traceback.format_exc()
        from desktop.paths import WORKSPACE_ROOT
        directory = WORKSPACE_ROOT / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "errors.log").open("a") as stream:
            stream.write(text + "\n")
        QMessageBox.critical(None, "Arknights Zero startup failed", text)
        return 1
    previous_hook = sys.excepthook
    def report_exception(kind, value, tb):
        window.show_error("".join(traceback.format_exception(kind, value, tb)))
    sys.excepthook = report_exception
    window.show()
    if args.config:
        window.training_view.load_config(args.config)
    if args.review_account:
        QTimer.singleShot(0, window.training_view._edit_account)
    if args.quit_after is not None:
        QTimer.singleShot(int(args.quit_after * 1000), window.begin_shutdown)
    result = app.exec()
    sys.excepthook = previous_hook
    return result


if __name__ == "__main__":
    raise SystemExit(main())
