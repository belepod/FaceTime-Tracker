# Installation Guide for Face Movement Tracker

This guide provides step-by-step instructions to set up and install the Face Movement Tracker project.

## 1. Prerequisites

*   **Python:** Ensure you have Python installed on your system (version 3.7 or higher is recommended). You can download it from [python.org](https://www.python.org/downloads/).
*   **Pip:** Python's package installer, pip, should be included with your Python installation.
*   **Git:** For cloning the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).
*   **Webcam:** A functional webcam is required for the real-time face tracking feature of `main.py`.

## 2. Clone the Repository

Open your terminal or command prompt and run the following command to clone the project repository:

```bash
git clone https://github.com/belepod/FaceTime-Tracker.git
cd FaceTime-Tracker
```

## 3. Set Up a Virtual Environment (Recommended)

Using a virtual environment helps manage project dependencies and avoids conflicts with other Python projects.

*   **Create a virtual environment:**
    Navigate to the project directory (`FaceTime-Tracker`) in your terminal and run:
    ```bash
    python -m venv venv
    ```
    This creates a folder named `venv` in your project directory.

*   **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    Your terminal prompt should now indicate that the virtual environment is active.

## 4. Install Dependencies

With the virtual environment activated, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This will install libraries such as OpenCV, face_recognition, pandas, matplotlib, and tensorflow.

*(Note: Installing some of these libraries, especially `dlib` (a dependency of `face_recognition`) and `tensorflow`, can sometimes be complex depending on your operating system and existing C++ compilers. If you encounter issues, please refer to the official installation guides for these packages for more detailed troubleshooting steps.)*

## 5. Webcam Setup

*   Ensure your webcam is connected and recognized by your operating system.
*   The `main.py` script will attempt to use the default webcam (usually index 0).
*   If you have multiple webcams or face issues, you might need to adjust the `cv2.VideoCapture(0)` line in `main.py` to use the correct camera index (e.g., `cv2.VideoCapture(1)`).

## 6. Image Path for Known Faces (for `main.py`)

The `main.py` script, in its current version, is configured to load known faces for recognition from a specific directory. You will need to:

1.  Create a directory on your system to store images of faces you want the program to recognize. For example, you could create `img` inside the project directory.
2.  Place clear images (e.g., JPG, PNG) of the individuals in this directory. Each image file should ideally be named after the person (e.g., `YourName.jpg`).
3.  **Crucially, you will need to update the `path` variable in `main.py`** to point to this directory.
    Currently, it is hardcoded as:
    ```python
    path = '/home/siddharth/Downloads/mlpbl/peh/img'
    ```
    Change this to the path of the directory you created. For example, if you created an `img` folder inside the project:
    ```python
    path = 'img' 
    # Or, for an absolute path:
    # path = '/path/to/your/project/FaceTime-Tracker/img'
    ```
    Additionally, `main.py` has a specific line to load "Siddharth's" image for encoding:
    ```python
    siddharth_image_path = next((f'{path}/{cl}' for cl in myList if cl.startswith('Siddharth')), None)
    ```
    If you are not "Siddharth" or want to recognize a different primary user, you'll need to adjust this logic or ensure an image named `Siddharth*.jpg` (or similar, matching the pattern) exists in your image path for the script to run as is. For recognizing multiple known people, the script would need further modification beyond this specific lookup.

You are now ready to run the application scripts. Refer to the main `README.md` for usage instructions.
