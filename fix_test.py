import sys
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, QCoreApplication

class WorkerSignals(QObject):
    finished = pyqtSignal(list)

class ProcessCleanupWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self):
        self.signals.finished.emit(["test"])

def test_run():
    app = QCoreApplication(sys.argv)
    worker = ProcessCleanupWorker()
    def on_finished(result):
        print("Finished:", result)
        app.quit()
    worker.signals.finished.connect(on_finished)
    QThreadPool.globalInstance().start(worker)
    app.exec()

test_run()
