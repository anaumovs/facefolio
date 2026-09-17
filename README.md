# FaceFolio

A local webcam application that recognizes consenting students and displays their profile alongside a captured face photo. Built with Python, OpenCV YuNet detection, SFace embeddings, PySide6 desktop widgets, NumPy matching, and SQLite storage.

## Quick start

Use Python 3.11 or newer. Run these commands from this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py download-models
python app.py
```

On Windows activate with `.venv\Scripts\activate`. On macOS allow Terminal camera access under System Settings → Privacy & Security → Camera. Close other camera apps if necessary, or try `python app.py recognize --camera 1`.

## Capture a student and create their profile

1. Show one consenting person's face in good light.
2. When the app shows **Unknown person**, click **Capture and add profile** or press **C**.
3. A guided enrollment window starts with that photo. Capture four more photos with the suggested small changes in angle, expression, and distance. Click **Capture photo (C)** or press **C** for each photo, waiting at least one second between captures. When all five are filled, **C** safely replaces only the fifth photo with a new capture. To redo the latest photo before that, use **Remove last photo (R)**, then press **C** to capture its replacement. Choose **Continue** when satisfied.
4. The form displays the face photo beside **First name**, **Last name**, and **Student ID**.
5. Enter both names and exactly seven digits for the student ID, then click **Save profile**. Invalid or duplicate IDs keep the form open and ask you to re-enter a valid ID. The ID field is highlighted and selected for correction; the names and captured photo remain in place. Nothing is saved until the entry passes validation. Leading zeroes are preserved, and invalid characters or extra digits are never silently removed.
6. The camera resumes recognizing the student directory. A confident match displays the saved photo, first name, last name, and student ID beside the live feed.

**Cancel** at either enrollment step discards the new captures and resumes the camera. Five photos must be completed before a new student profile can be saved. A similarity check against the first capture helps flag a changed person or an excessive face angle; it is not proof of identity. Photos are written only when saving a profile or explicitly replacing an existing enrollment set. When a registered student matches, the same button changes to **Update 5 profile photos (C)**; clicking it or pressing **C** opens enrollment for that existing student. Capture remains disabled when image quality is insufficient or when zero or multiple faces are visible. The portrait and face embedding come from the same unannotated frame.

Click the camera or capture window before using its keyboard shortcuts. In the retake popup, **C** and **Shift+C** work even when Capture, Remove, or Cancel has focus. Holding **C** does not take repeated photos. Close the window or press **Q** in the live camera tab to quit. Keep the form open while entering names: keyboard shortcuts for the main camera window are inactive while the form is modal.

## Student directory

Open the **Student directory** tab to search by first name, last name, or seven-digit student ID. Select a row to view its enrollment photos.

- **Edit details** updates the names and ID while preserving the face data. Invalid and duplicate IDs keep the form open for correction.
- **Replace with 5 photos (C)** upgrades an older single-photo profile or refreshes an existing enrollment. Select the student row and press **C** (or click the button), then keep that student in view. Typing **C** in the search field just types a letter. The previous set stays intact unless five new photos are captured and **Save 5 photos** is selected.
- **Delete profile** asks you to confirm the selected name and ID, then removes the profile and all of its enrollment photos. Cancel leaves it unchanged.

The live matching gallery refreshes after each change. Camera recognition pauses while viewing the directory. The main camera shortcuts are disabled in those tabs so typing a name cannot close the app; the directory’s table has its own **C** shortcut for the selected student.

## Existing profiles and management

The database is upgraded automatically without deleting old data. Earlier command-line demo profiles did not include student IDs or photos. They remain stored and can be verified by ID, but are outside the new student directory. This prevents repeated earlier demo enrollments from competing with a student's new profile. Add a student profile through the capture form to include that person in the directory.

```sh
python app.py profiles
python app.py verify --id PROFILE_ID
python app.py delete PROFILE_ID
```

`profiles` lists saved details without printing face embeddings or photo data. `verify` compares the webcam face against a selected profile. `delete` removes the profile and its photo from the active database. In verification mode, adding a new profile switches the camera back to recognition of the student directory.

The earlier five-sample command-line enrollment remains available:

```sh
python app.py enroll --name "Alex" --bio "Demo participant" --consent
```

Press Space five times with slightly different angles and expressions, at least one second apart. After the fifth capture it saves a legacy profile and stays open to verify it. This earlier command does not create a student profile or save a photo; use the desktop capture form for those fields.

## How it works

Camera frame → YuNet face and landmarks → quality gate → aligned face → SFace embedding → cosine comparison → threshold and competing-match margin → profile lookup or unknown.

New student enrollments store five normalized embeddings and five JPEG face crops. The first photo is the profile portrait. Earlier single-photo student profiles remain usable and can be upgraded in the directory. The matcher averages up to the three strongest sample similarities, using however many samples actually exist. Multiple samples provide broader reference coverage, but any accuracy improvement must be measured on held-out images.

Identification requires a score of at least 0.55 and a lead of at least 0.08 over the second-best candidate. Both are configurable, for example `python app.py recognize --threshold 0.6 --margin 0.1`. These are starting values, not calibrated accuracy claims. Similarity is not a probability. Live decisions are independent per frame, so matches can flicker. Different student IDs assigned to the same face can still produce ambiguous matches; IDs are unique, but face uniqueness is not guaranteed.

All profile text in the desktop app is rendered as plain text and supports Unicode. Database writes use parameters. The seven-digit ID rule is checked in both the form and the storage layer, with a unique database index to prevent collisions.

## Measure accuracy from the terminal

Keep enrollment, threshold-development, and final test captures separate, preferably recorded in different sessions. Include consenting people who are not enrolled. Test lighting, camera distances, glasses, and face angles. Never report pretrained model benchmark scores as this application's accuracy.

Create `private/test.csv` with your own held-out images:

```csv
path,expected_id
images/alex-session2.jpg,PASTE_ACTUAL_PROFILE_UUID
images/unenrolled-volunteer.jpg,
```

`expected_id` is the profile UUID printed by `python app.py profiles`, not the seven-digit student ID. Image paths are relative to the CSV. Run:

```sh
python evaluate.py private/test.csv --threshold 0.55 --margin 0.08
```

By default, evaluation uses the same student-only gallery as the desktop directory. Pass `--gallery all` to include legacy demo enrollments when evaluating those profiles.

This optional command-line evaluator reports correct decisions, known-person match/miss rates, unknown false matches, capture failures, and processing times. Missing groups produce null rates rather than misleading zeroes. Tune thresholds on development captures, then freeze them before the final test. Report participant counts, image counts, gallery size, rates, and capture conditions. Small samples cannot substantiate high-accuracy claims. This evaluator measures gallery identification; selected-profile verification needs a separate genuine/impostor pair evaluation before reporting verification metrics.

## Data and limitations

Enrollment photos, embeddings, names, student IDs, biographies from older profiles, enrollment timestamps, remain in `private/profiles.sqlite3`. Video is not recorded. A candidate capture stays in memory until the form is saved or cancelled. Embeddings are sensitive biometric data, not anonymous hashes. Storage is not encrypted; filesystem permissions limit access where supported. Deletion removes the active database record and stored photo but does not erase independent backups or guarantee forensic deletion. Saving confirms the operator is enrolling a consenting participant; the timestamp is not a signed consent form.

The application does not implement liveness or presentation-attack detection and can be fooled by a photograph or video. It is a portfolio demonstration, not a secure login system. All inference is local after downloading dependencies and models. Do not include actual participant information or test images in GitHub commits or screenshots without permission. `.gitignore` excludes private data, downloaded models, and the virtual environment.

## Tests and project layout

```sh
python -m unittest discover -s tests -v
```

The automated tests use synthetic vectors and a simulated camera. They cover mouse clicks held across camera refreshes, actual C/R key events, safe retakes, shortcut isolation while typing, guided five-photo enrollment, ID validation, duplicate IDs, atomic photo replacement, profile search/edit/delete, cascading photo deletion, exact-photo leakage checks, database migration, camera failures, and legacy enrollment. They do not establish real-world biometric accuracy. GUI tests render offscreen and never open the physical webcam.

- `app.py`: entry point, model loading, earlier webcam enrollment, and CLI
- `desktop.py`: application tabs, live camera, and shared camera lifecycle
- `enrollment.py`: guided five-photo capture and retakes
- `profile_form.py`: validated profile creation and editing
- `directory.py`: searchable student directory and photo management
- `ui.py`: shared widgets, style, and image conversion
- `core.py`: validation, normalization, matching, database migration, and persistence
- `evaluation.py`: shared held-out evaluation and metric definitions
- `evaluate.py`: command-line evaluation
- `tests/`: matching, storage, enrollment, and desktop workflow tests
- `.github/workflows/tests.yml`: automated tests for GitHub

## Model sources and attribution

The pretrained networks are provided by [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Review the model-specific licenses for [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) before redistributing model files. The downloader retrieves upstream model filenames from the main branch; model checksum pinning is a future reproducibility improvement. `requirements-lock.txt` records versions used for local verification; install it with `pip install -r requirements-lock.txt` to reproduce those versions.

The face pipeline follows the [OpenCV face detection and recognition API tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html). The desktop interface uses [Qt for Python widgets](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html). This project integrates pretrained models; it does not train its own network.

## Portfolio presentation

Demonstrate an unknown face, guided five-photo enrollment, recognition, directory search/edit/delete, invalid-ID rejection, and a command-line evaluation with independently captured test photos. Explain threshold/margin tradeoffs, database migration and constraints, and false-match evaluation. Include measured results and a consented demo recording before claiming accuracy on a resume.

Suggested resume wording after running and understanding the project: “Built a local student-profile recognition application using Python, OpenCV, PySide6, and SQLite, with webcam enrollment, validated student IDs, photo profiles, open-set rejection, and automated workflow tests.” Add measured performance only after evaluation.
