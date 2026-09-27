# Security

The project processes untrusted media files and untrusted engine containers.

- Report vulnerabilities privately via GitHub security advisories on this repository.
- Engine containers run without network access and with read-only mounts of the audio.
- Media decoding goes through ffmpeg in a subprocess with resource limits.
- Credentials (Hugging Face, Porcupine access keys) are never committed; see `.gitignore`.
