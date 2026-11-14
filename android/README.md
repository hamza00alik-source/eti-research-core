# ETI Android shell

This is the first native Android packaging layer for ETI. It securely opens the existing HTTPS ETI research PWA inside an Android WebView and routes off-domain links to the system browser.

- Minimum Android version: API 23
- Target SDK: 35
- No exchange credentials are embedded in the APK.
- Cleartext traffic, local file access, content access, multiple windows and mixed content are disabled.
- Only the ETI Railway HTTPS host is loaded inside the WebView.

**Important:** this is a native app shell, not yet an offline/on-device AI client. ONNX model inference, local feature generation, signed release packaging and device-level tests are separate milestones. The debug APK is suitable for controlled installation/testing only, not public release.


The Android project also contains an ONNX Runtime inference wrapper. It expects a matching reviewed `eti_model.onnx` and `eti_model_metadata.json` in app assets. These files are not checked into the repository; they are generated only after a candidate model is trained and exported. The wrapper validates input width and finite values, uses the exact feature order from metadata, and applies ETI temperature scaling after ONNX inference. A local OHLCV-to-feature implementation and physical-device parity test are still required before offline AI can be enabled.
