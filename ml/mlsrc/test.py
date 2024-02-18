import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Check if the event is not for a directory
            print(f"File created: {event.src_path}")

def monitor_directory(directory):
    event_handler = FileCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__test__":
    directory_to_watch = "/home/alex/Documents/ml/ml/test"  # Specify the directory you want to monitor
    monitor_directory(directory_to_watch)
