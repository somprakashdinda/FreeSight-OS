# Android Gaze Controller — corrected build

This folder is a corrected copy of the supplied project. The original project has not been changed.

## Before running

1. Install Python 3.11 and add it to `PATH`.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Install Android Platform Tools (`adb`) and scrcpy, and enable USB debugging on the phone.
4. Verify the phone with `adb devices`.
5. Run `python android_interop.py` first. Only proceed to `python main.py` after the camera diagnostic receives a frame.

The controller starts in terminal-only mode. Use directional scrolling first; use precision clicking only after completing calibration and validating its accuracy.

## Important hardware note

The supplied design uses scrcpy's internal camera/control protocol. Its options vary by scrcpy release and phone manufacturer. Use a current matching scrcpy client/server pair. If the camera stream fails, test the device's Android USB Webcam/UVC mode separately before changing application code.

## Corrected defects

- repaired syntax-breaking indentation in `android_interop.py`;
- restored Android input and frame-normalization methods to `AndroidConnector`;
- replaced the inconsistent calibration mapper with a consistent, regularized implementation;
- aligned pose-aware prediction with the mapper API;
- restored all fields required by `SystemState.get_snapshot()`;
- connected iris-normalized coordinates to the vision pipeline;
- corrected the head-pose adapter to call `HeadPoseEstimator.estimate()`.
- corrected portrait camera-frame normalization (1080x1920, matching the
  requested 9:16 front-camera stream);
- made calibration profiles reject incompatible device sizes or non-finite
  coefficients, and enabled the pose-aware model for complete nine-point
  calibrations.
