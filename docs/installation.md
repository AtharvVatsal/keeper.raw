# Installation Guide

## Windows

### Step 1: Download keeper.raw

Go to the [Releases page](https://github.com/AtharvVatsal/keeper.raw/releases/latest) and download `keeper-raw-x.x.x-x64-setup.exe`.

### Step 2: Install ExifTool

keeper.raw requires ExifTool for RAW preview extraction.

1. Go to https://exiftool.org/
2. Download the **Windows Executable** (the standalone `.exe` version)
3. Extract `exiftool(-k).exe` from the zip
4. Rename it to `exiftool.exe`
5. Move it to a folder on your PATH (e.g., `C:\Windows\` or create `C:\Tools\` and add it to PATH)

To verify it's installed, open PowerShell and run:
```powershell
exiftool -ver
```

You should see a version number like `12.97`.

### Step 3: Run keeper.raw

Double-click the installer and follow the prompts. That's it.

---

## macOS

### Step 1: Download keeper.raw

Go to the [Releases page](https://github.com/AtharvVatsal/keeper.raw/releases/latest) and download:
- `keeper-raw-x.x.x-aarch64.dmg` for Apple Silicon (M1/M2/M3/M4)
- `keeper-raw-x.x.x-x64.dmg` for Intel Macs

### Step 2: Install ExifTool
```bash
brew install exiftool
```

Or download from https://exiftool.org/ and follow the macOS instructions.

### Step 3: Run keeper.raw

Open the `.dmg`, drag keeper.raw to your Applications folder, and launch it.

> **Note:** On first launch, macOS may show a security warning. Go to System Settings → Privacy & Security and click "Open Anyway."

---

## Linux

### Step 1: Download keeper.raw

Go to the [Releases page](https://github.com/AtharvVatsal/keeper.raw/releases/latest) and download:
- `keeper-raw-x.x.x-amd64.AppImage` (works on most distros)
- `keeper-raw-x.x.x-amd64.deb` (Debian/Ubuntu)

### Step 2: Install ExifTool
```bash
# Debian/Ubuntu
sudo apt install libimage-exiftool-perl

# Fedora
sudo dnf install perl-Image-ExifTool

# Arch
sudo pacman -S perl-image-exiftool
```

### Step 3: Run keeper.raw

For AppImage:
```bash
chmod +x keeper-raw-x.x.x-amd64.AppImage
./keeper-raw-x.x.x-amd64.AppImage
```

For .deb:
```bash
sudo dpkg -i keeper-raw-x.x.x-amd64.deb
keeper-raw
```