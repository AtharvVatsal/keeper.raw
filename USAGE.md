# User Guide

A comprehensive guide to using keeper-raw for photo culling.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [The Culling Workflow](#the-culling-workflow)
3. [Understanding the Interface](#understanding-the-interface)
4. [Keyboard Shortcuts](#keyboard-shortcuts)
5. [Interpreting Results](#interpreting-results)
6. [Exporting to Editing Software](#exporting-to-editing-software)
7. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

### First Launch

1. Open keeper-raw
2. You'll see the main window with the header and an empty grid

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, macOS 11, Linux | Latest |
| RAM | 8 GB | 16 GB |
| Storage | 500 MB | 1 GB |
| ExifTool | Installed | Latest |

---

## The Culling Workflow

### Step 1: Open Your Photos

Click **"Open Folder"** and navigate to the folder containing your RAW photos.

![Open Folder Button](docs/screenshots/screenshot_cull.png)

keeper-raw will:
- Scan for supported RAW files
- Extract embedded JPEG previews
- Display thumbnails in the grid

> **Note**: If some photos don't show previews, they may not contain embedded JPEGs. Try converting to DNG or using a different camera.

### Step 2: Run AI Culling

Click **"Cull Images"** to start the analysis.

The AI will:
1. **Group photos** into scenes (bursts taken together)
2. **Detect faces** in each photo
3. **Analyze sharpness** using Laplacian variance
4. **Detect blinks** using Eye Aspect Ratio
5. **Select keepers** based on sharpness and blink status

A progress indicator shows how many images have been processed.

### Step 3: Review Results

Browse the grid to see:
- **Green border**: AI-selected keeper
- **Red border**: Rejected (blink detected)
- **No border**: Unrated

Click any image to open the **Loupe View** for detailed inspection.

### Step 4: Make Adjustments

In the Loupe View or grid:
- Press `K` to mark as keeper
- Press `X` to mark as reject  
- Press `U` to remove rating

### Step 5: Export

Click **"Export XMP"** to save your ratings as XMP sidecar files.

---

## Understanding the Interface

### Main Window

```
┌────────────────────────────────────────────────────────────┐
│  keeper.raw                                    ⚙ Settings │
├────────────────────────────────────────────────────────────┤
│  [Open Folder] [Cull Images] [Export XMP]                 │
│                                                            │
│  📁 /path/to/photos                                      │
│  127 images loaded ─ ready to cull                       │
│                                                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │    ★    │ │         │ │         │ │    X    │        │
│  │ KEEPER  │ │         │ │         │ │ REJECT  │  ...   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
│                                                            │
│  [All (42)] [Keepers (38)] [Rejects (4)]                 │
└────────────────────────────────────────────────────────────┘
```

### Scene Card

- **Single image**: Shows just that image
- **Burst**: Shows primary image with a blue badge showing count (e.g., "5")
- Click badge to expand and see all images in the burst

### Loupe View

Detailed inspection view with:
- Large image display
- Face-centered zoom (if face detected)
- Image metadata
- Score information
- Quick rating buttons

---

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| `←` | Previous scene |
| `→` | Next scene |
| `↑` | Previous image in scene |
| `↓` | Next image in scene |
| `Esc` | Close loupe / Close settings |

### Rating

| Key | Action |
|-----|--------|
| `K` | Mark as keeper |
| `X` | Toggle reject |
| `U` | Unrate (remove rating) |

### View

| Key | Action |
|-----|--------|
| `Z` | Toggle zoom in loupe |
| `Space` | (Reserved) |

---

## Interpreting Results

### What Makes a Good Keeper?

The AI selects keepers based on:

1. **Sharpness**: Higher Laplacian variance = sharper
2. **Face Detection**: Prioritizes images with detected faces
3. **Blink Status**: Prefers images without closed eyes

### Understanding Scores

In the Loupe View, you'll see:

- **Sharpness**: Numeric score (higher = sharper)
- **Face detected**: Yes/No
- **Blink detected**: Yes/No

### When AI Might Be Wrong

The AI isn't perfect. Watch for:

- **Motion blur**: Fast-moving subjects may appear sharp but be unusable
- **Poor lighting**: High ISO noise can affect sharpness scores
- **Unusual poses**: Some poses may confuse face detection
- **Group photos**: AI picks best overall, not best for each person

---

## Exporting to Editing Software

### Supported Software

| Software | Version | Notes |
|----------|---------|-------|
| Lightroom Classic | CC 2015+ | Full support |
| Lightroom (cloud) | Latest | Full support |
| Darktable | 4.x+ | Full support |
| Capture One | 23+ | Full support |

### How Export Works

1. keeper-raw creates `.xmp` files next to your RAW files
2. The editing software reads these when you open the folder
3. Ratings and labels appear in the software

### Example

```
Before export:
  /photos/
    DSC_0571.NEF
    DSC_0572.NEF

After export:
  /photos/
    DSC_0571.NEF
    DSC_0571.xmp    ← Created by keeper-raw
    DSC_0572.NEF
    DSC_0572.xmp    ← Created by keeper-raw
```

### Opening in Lightroom

1. Click "Export XMP" in keeper-raw
2. Open Lightroom
3. Go to File → Open Folder
4. Select your photo folder
5. Your ratings will appear!

---

## Tips and Best Practices

### Before Culling

1. **Organize first**: Put all photos from one shoot in one folder
2. **Check previews**: Make sure RAW files have embedded JPEGs
3. **Close other apps**: Free up RAM for processing

### During Culling

1. **Use keyboard shortcuts**: Much faster than clicking
2. **Start with Keepers view**: See what the AI chose
3. **Check Rejects**: Make sure nothing good was rejected
4. **Use loupe**: Don't rely solely on thumbnails

### After Culling

1. **Export immediately**: Don't skip this step!
2. **Backup XMP files**: They're just text, easy to lose
3. **Test in editing software**: Verify ratings appear correctly

### Fine-Tuning Settings

If results aren't what you expect:

| Problem | Solution |
|---------|----------|
| Too few keepers | Increase burst threshold |
| Too many keepers | Decrease blink sensitivity |
| Wrong selections | Increase face confidence |
| Split scenes incorrectly | Adjust visual similarity |

---

## Troubleshooting

### "No images found"

- Check that folder contains supported RAW formats
- Verify ExifTool is installed and in PATH

### "No embedded preview found"

- Some RAW files don't have JPEGs
- Try converting to DNG first

### "Export didn't work"

- Check folder is writable
- Verify XMP files were created
- Try running editing software as administrator

### Slow performance

- Reduce batch size
- Close other applications
- Ensure adequate RAM

---

## Getting Help

- Check [README.md](README.md) for setup help
- See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Open an issue for bugs
- Start a discussion for questions

---

Happy culling! 📷
