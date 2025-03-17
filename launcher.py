import os
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser


class StreamlitLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Percentile Calculator")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f2f6")  # Streamlit-like color

        # Center content
        self.frame = tk.Frame(root, bg="#f0f2f6")
        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # App title
        self.title_label = tk.Label(
            self.frame,
            text="Percentile Calculator",
            font=("Arial", 18, "bold"),
            bg="#f0f2f6",
            fg="#0e1117",
        )
        self.title_label.pack(pady=10)

        # Launch button
        self.launch_button = tk.Button(
            self.frame,
            text="Launch Web App",
            command=self.launch_streamlit,
            bg="#ff4b4b",  # Streamlit primary color
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
        )
        self.launch_button.pack(pady=20)

        # Status label
        self.status_label = tk.Label(
            self.frame,
            text="Ready to launch",
            font=("Arial", 10),
            bg="#f0f2f6",
            fg="#0e1117",
        )
        self.status_label.pack(pady=5)

        # Server process
        self.process = None

    def launch_streamlit(self):
        if self.process is None:
            self.status_label.config(text="Starting server...")
            threading.Thread(target=self.run_server, daemon=True).start()
        else:
            self.status_label.config(text="Server is already running")
            # Open browser again if server is already running
            webbrowser.open("http://localhost:8501")

    def run_server(self):
        # Path to your Streamlit app
        app_path = "grids/streamlit_app.py"

        # If packaged with PyInstaller, adjust path
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
            app_path = os.path.join(app_dir, "app", app_path)

        # Launch Streamlit
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                "--server.headless",
                "true",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(2)

        # Update UI
        self.root.after(
            0, self.update_status, "Server running at http://localhost:8501"
        )

        # Open web browser
        webbrowser.open("http://localhost:8501")

    def update_status(self, text):
        self.status_label.config(text=text)

    def on_closing(self):
        if self.process:
            self.process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StreamlitLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
