import { useState, useMemo, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

interface ImageSummary {
  id: number;
  file_name: string;
  timestamp: number | null;
  camera_model: string | null;
  focal_length_mm: number | null;
  iso: number | null;
  preview_width: number;
  preview_height: number;
  preview_path: string;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

type Verdict = "Keeper" | { Reject: { reason: string } } | "Unrated";

interface ImageScore {
  image_id: number;
  sharpness_score: number;
  has_face: boolean;
  face_bbox: BoundingBox | null;
  blink_detected: boolean;
  eye_aspect_ratio: number | null;
  verdict: Verdict;
}

interface Scene {
  id: number;
  image_ids: number[];
  keeper_id: number | null;
}

interface CullManifest {
  total_images: number;
  total_scenes: number;
  total_keepers: number;
  total_rejects: number;
  total_unrated: number;
  scenes: Scene[];
  image_scores: ImageScore[];
  processing_time_secs: number;
}

interface ExportResponse {
  files_written: number;
  files_skipped: number;
  errors: string[];
}

interface CullConfig {
  burst_threshold_secs: number;
  phash_similarity_threshold: number;
  blink_ear_threshold: number;
  face_detection_confidence: number;
}

type OverrideVerdict = "keeper" | "reject" | "unrated";
type OverrideMap = Map<number, OverrideVerdict>;
function getVerdictLabel(v: Verdict): "keeper" | "reject" | "unrated" {
  if (v === "Keeper") return "keeper";
  if (typeof v === "object" && "Reject" in v) return "reject";
  return "unrated";
}

function getEffectiveVerdict(
  imageId: number,
  scoreMap: Map<number, ImageScore>,
  overrides: OverrideMap
): "keeper" | "reject" | "unrated" {
  const override = overrides.get(imageId);
  if (override) return override;
  const score = scoreMap.get(imageId);
  if (!score) return "unrated";
  return getVerdictLabel(score.verdict);
}

function getEffectiveKeeperId(
  scene: Scene,
  overrides: OverrideMap
): number | null {
  for (const id of scene.image_ids) {
    if (overrides.get(id) === "keeper") return id;
  }
  if (scene.keeper_id != null) {
    const override = overrides.get(scene.keeper_id);
    if (override === "unrated" || override === "reject") return null;
  }
  return scene.keeper_id;
}

function getSortedSceneImageIds(
  scene: Scene,
  scoreMap: Map<number, ImageScore>,
  overrides: OverrideMap
): number[] {
  const effectiveKeeperId = getEffectiveKeeperId(scene, overrides);
  return [...scene.image_ids].sort((a, b) => {
    if (a === effectiveKeeperId) return -1;
    if (b === effectiveKeeperId) return 1;
    const scoreA = scoreMap.get(a)?.sharpness_score ?? 0;
    const scoreB = scoreMap.get(b)?.sharpness_score ?? 0;
    return scoreB - scoreA;
  });
}

function SettingsPanel({
  config,
  onSave,
  onClose,
}: {
  config: CullConfig;
  onSave: (config: CullConfig) => void;
  onClose: () => void;
}) {
  // Local state so changes don't apply until "Save"
  const [burstThreshold, setBurstThreshold] = useState(config.burst_threshold_secs);
  const [phashThreshold, setPhashThreshold] = useState(config.phash_similarity_threshold);
  const [earThreshold, setEarThreshold] = useState(config.blink_ear_threshold);
  const [faceConfidence, setFaceConfidence] = useState(config.face_detection_confidence);

  function handleSave() {
    onSave({
      burst_threshold_secs: burstThreshold,
      phash_similarity_threshold: phashThreshold,
      blink_ear_threshold: earThreshold,
      face_detection_confidence: faceConfidence,
    });
  }

  function handleReset() {
    setBurstThreshold(3.0);
    setPhashThreshold(14);
    setEarThreshold(0.21);
    setFaceConfidence(0.5);
  }

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const sliderStyle = {
    width: "100%",
    accentColor: "#f59e0b",
    cursor: "pointer",
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.85)",
        zIndex: 2000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          backgroundColor: "#1a1a1a",
          borderRadius: 12,
          border: "1px solid #333",
          padding: 32,
          width: 480,
          maxHeight: "90vh",
          overflowY: "auto",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 24,
          }}
        >
          <h2 style={{ margin: 0, fontSize: 20, color: "#fff" }}>
            Settings
          </h2>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "1px solid #555",
              color: "#aaa",
              padding: "4px 12px",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            ✕
          </button>
        </div>

        {/* ============================================================
            BURST THRESHOLD
            
            How many seconds between consecutive shots to consider
            them part of the same burst. Lower = stricter grouping
            (only rapid-fire bursts), higher = more permissive
            (groups shots taken a few seconds apart).
            ============================================================ */}
        <div style={{ marginBottom: 24 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 6,
            }}
          >
            <label style={{ fontSize: 14, color: "#ccc" }}>
              Burst Threshold
            </label>
            <span style={{ fontSize: 14, color: "#f59e0b", fontWeight: "bold" }}>
              {burstThreshold.toFixed(1)}s
            </span>
          </div>
          <input
            type="range"
            min="0.5"
            max="10"
            step="0.5"
            value={burstThreshold}
            onChange={(e) => setBurstThreshold(parseFloat(e.target.value))}
            style={sliderStyle}
          />
          <p style={{ fontSize: 11, color: "#666", margin: "4px 0 0 0" }}>
            Max seconds between shots to group as a burst. Lower = only
            rapid-fire bursts. Higher = groups shots taken further apart.
          </p>
        </div>

        {/* ============================================================
            PHASH SIMILARITY
            
            How visually similar two images must be to stay in the
            same scene. Lower = stricter (only nearly identical frames),
            higher = more forgiving (allows subject movement).
            ============================================================ */}
        <div style={{ marginBottom: 24 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 6,
            }}
          >
            <label style={{ fontSize: 14, color: "#ccc" }}>
              Visual Similarity
            </label>
            <span style={{ fontSize: 14, color: "#f59e0b", fontWeight: "bold" }}>
              {phashThreshold}
            </span>
          </div>
          <input
            type="range"
            min="4"
            max="30"
            step="1"
            value={phashThreshold}
            onChange={(e) => setPhashThreshold(parseInt(e.target.value))}
            style={sliderStyle}
          />
          <p style={{ fontSize: 11, color: "#666", margin: "4px 0 0 0" }}>
            Max visual difference (Hamming distance) to keep images in the
            same scene. Lower = stricter matching. Higher = more forgiving.
          </p>
        </div>

        {/* ============================================================
            BLINK THRESHOLD (EAR)
            
            Eye Aspect Ratio below which a blink is detected. Lower =
            only catches fully closed eyes. Higher = catches squinting
            and half-blinks too (but more false positives).
            ============================================================ */}
        <div style={{ marginBottom: 24 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 6,
            }}
          >
            <label style={{ fontSize: 14, color: "#ccc" }}>
              Blink Sensitivity
            </label>
            <span style={{ fontSize: 14, color: "#f59e0b", fontWeight: "bold" }}>
              {earThreshold.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.35"
            step="0.01"
            value={earThreshold}
            onChange={(e) => setEarThreshold(parseFloat(e.target.value))}
            style={sliderStyle}
          />
          <p style={{ fontSize: 11, color: "#666", margin: "4px 0 0 0" }}>
            Eye Aspect Ratio threshold. Lower = only flags fully closed
            eyes. Higher = catches squinting too (more false positives).
          </p>
        </div>

        {/* ============================================================
            FACE DETECTION CONFIDENCE
            
            How confident the YOLOv8 model must be that it found a face.
            Lower = detects more faces (including partial/angled ones)
            but with more false positives. Higher = only high-confidence
            detections.
            ============================================================ */}
        <div style={{ marginBottom: 28 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 6,
            }}
          >
            <label style={{ fontSize: 14, color: "#ccc" }}>
              Face Detection Confidence
            </label>
            <span style={{ fontSize: 14, color: "#f59e0b", fontWeight: "bold" }}>
              {(faceConfidence * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.20"
            max="0.90"
            step="0.05"
            value={faceConfidence}
            onChange={(e) => setFaceConfidence(parseFloat(e.target.value))}
            style={sliderStyle}
          />
          <p style={{ fontSize: 11, color: "#666", margin: "4px 0 0 0" }}>
            Min confidence for face detection. Lower = detects more faces
            (including partial). Higher = only high-confidence detections.
          </p>
        </div>

        {/* Buttons */}
        <div
          style={{
            display: "flex",
            gap: 12,
            justifyContent: "space-between",
          }}
        >
          <button
            onClick={handleReset}
            style={{
              padding: "8px 20px",
              fontSize: 13,
              cursor: "pointer",
              backgroundColor: "transparent",
              color: "#888",
              border: "1px solid #444",
              borderRadius: 4,
            }}
          >
            Reset to Defaults
          </button>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={onClose}
              style={{
                padding: "8px 20px",
                fontSize: 13,
                cursor: "pointer",
                backgroundColor: "transparent",
                color: "#aaa",
                border: "1px solid #555",
                borderRadius: 4,
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              style={{
                padding: "8px 20px",
                fontSize: 13,
                cursor: "pointer",
                backgroundColor: "#f59e0b",
                color: "#000",
                border: "none",
                borderRadius: 4,
                fontWeight: "bold",
              }}
            >
              Save Settings
            </button>
          </div>
        </div>

        <p
          style={{
            fontSize: 11,
            color: "#555",
            margin: "16px 0 0 0",
            textAlign: "center",
          }}
        >
          Changes take effect on the next cull. Re-cull to see updated results.
        </p>
      </div>
    </div>
  );
}

function LoupeView({
  image,
  score,
  sceneLabel,
  imageIndexLabel,
  effectiveVerdict,
  onClose,
  onPrevScene,
  onNextScene,
  onPrevImage,
  onNextImage,
  onMakeKeeper,
  onUnrate,
  onToggleReject,
}: {
  image: ImageSummary;
  score: ImageScore | undefined;
  sceneLabel: string;
  imageIndexLabel: string;
  effectiveVerdict: "keeper" | "reject" | "unrated";
  onClose: () => void;
  onPrevScene: () => void;
  onNextScene: () => void;
  onPrevImage: () => void;
  onNextImage: () => void;
  onMakeKeeper: () => void;
  onUnrate: () => void;
  onToggleReject: () => void;
}) {
  const [zoomed, setZoomed] = useState(true);

  useEffect(() => {
    setZoomed(true);
  }, [image.id]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape":
          onClose();
          break;
        case "ArrowLeft":
          e.preventDefault();
          onPrevScene();
          break;
        case "ArrowRight":
          e.preventDefault();
          onNextScene();
          break;
        case "ArrowUp":
          e.preventDefault();
          onPrevImage();
          break;
        case "ArrowDown":
          e.preventDefault();
          onNextImage();
          break;
        case "z":
        case "Z":
          setZoomed((prev) => !prev);
          break;
        case "k":
        case "K":
          onMakeKeeper();
          break;
        case "u":
        case "U":
          onUnrate();
          break;
        case "x":
        case "X":
          onToggleReject();
          break;
      }
    },
    [
      onClose,
      onPrevScene,
      onNextScene,
      onPrevImage,
      onNextImage,
      onMakeKeeper,
      onUnrate,
      onToggleReject,
    ]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const zoomStyle = useMemo(() => {
    if (!zoomed) {
      return {
        width: "100%",
        height: "100%",
        objectFit: "contain" as const,
        position: "relative" as const,
        left: 0,
        top: 0,
      };
    }

    const viewW = window.innerWidth;
    const viewH = window.innerHeight;
    let faceCenterX: number;
    let faceCenterY: number;

    if (score?.face_bbox) {
      const bb = score.face_bbox;
      faceCenterX = bb.x + bb.width / 2;
      faceCenterY = bb.y + bb.height / 2;
    } else {
      faceCenterX = image.preview_width / 2;
      faceCenterY = image.preview_height / 2;
    }

    const left = viewW / 2 - faceCenterX;
    const top = viewH / 2 - faceCenterY;

    return {
      width: image.preview_width,
      height: image.preview_height,
      objectFit: "none" as const,
      position: "absolute" as const,
      left,
      top,
    };
  }, [zoomed, image, score]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.95)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* ====== TOP BAR ====== */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 20px",
          backgroundColor: "rgba(0, 0, 0, 0.8)",
          zIndex: 1001,
          flexShrink: 0,
        }}
      >
        {/* Left: file info */}
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span
              style={{
                fontSize: 15,
                fontWeight: "bold",
                color: "#fff",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {image.file_name}
            </span>

            {effectiveVerdict === "keeper" && (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: "bold",
                  color: "#22c55e",
                  border: "1px solid #22c55e",
                  padding: "1px 6px",
                  borderRadius: 3,
                  flexShrink: 0,
                }}
              >
                ★ KEEPER
              </span>
            )}
            {effectiveVerdict === "reject" && (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: "bold",
                  color: "#ef4444",
                  border: "1px solid #ef4444",
                  padding: "1px 6px",
                  borderRadius: 3,
                  flexShrink: 0,
                }}
              >
                REJECT
              </span>
            )}
          </div>

          <p style={{ margin: "3px 0 0 0", fontSize: 11, color: "#888" }}>
            {image.camera_model || "Unknown camera"}
            {image.iso ? ` · ISO ${image.iso}` : ""}
            {image.focal_length_mm ? ` · ${image.focal_length_mm}mm` : ""}
            {score
              ? ` · Sharpness: ${score.sharpness_score.toFixed(1)}`
              : ""}
            {score?.face_bbox ? " · Face detected" : ""}
          </p>
        </div>

        {/* Center: override controls */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexShrink: 0,
            margin: "0 16px",
          }}
        >
          {effectiveVerdict !== "keeper" && (
            <button
              onClick={onMakeKeeper}
              title="K"
              style={{
                padding: "5px 12px",
                fontSize: 12,
                fontWeight: "bold",
                cursor: "pointer",
                backgroundColor: "rgba(34, 197, 94, 0.15)",
                color: "#22c55e",
                border: "1px solid #22c55e",
                borderRadius: 4,
              }}
            >
              ★ Keeper (K)
            </button>
          )}

          {effectiveVerdict === "keeper" && (
            <button
              onClick={onUnrate}
              title="U"
              style={{
                padding: "5px 12px",
                fontSize: 12,
                fontWeight: "bold",
                cursor: "pointer",
                backgroundColor: "rgba(255, 255, 255, 0.05)",
                color: "#aaa",
                border: "1px solid #555",
                borderRadius: 4,
              }}
            >
              Unrate (U)
            </button>
          )}

          {effectiveVerdict !== "reject" && (
            <button
              onClick={onToggleReject}
              title="X"
              style={{
                padding: "5px 12px",
                fontSize: 12,
                fontWeight: "bold",
                cursor: "pointer",
                backgroundColor: "rgba(239, 68, 68, 0.15)",
                color: "#ef4444",
                border: "1px solid #ef4444",
                borderRadius: 4,
              }}
            >
              Reject (X)
            </button>
          )}

          {effectiveVerdict === "reject" && (
            <button
              onClick={onToggleReject}
              title="X"
              style={{
                padding: "5px 12px",
                fontSize: 12,
                fontWeight: "bold",
                cursor: "pointer",
                backgroundColor: "rgba(255, 255, 255, 0.05)",
                color: "#aaa",
                border: "1px solid #555",
                borderRadius: 4,
              }}
            >
              Unreject (X)
            </button>
          )}
        </div>

        {/* Right: scene info and close */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 12, color: "#666" }}>
            {sceneLabel} · {imageIndexLabel}
          </span>

          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "1px solid #555",
              color: "#aaa",
              padding: "6px 14px",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* ====== IMAGE AREA ====== */}
      <div
        onClick={() => setZoomed((prev) => !prev)}
        style={{
          flex: 1,
          position: "relative",
          overflow: "hidden",
          cursor: zoomed ? "zoom-out" : "zoom-in",
        }}
      >
        <img
          src={convertFileSrc(image.preview_path)}
          alt={image.file_name}
          style={zoomStyle}
        />
      </div>

      {/* ====== BOTTOM HINT BAR ====== */}
      <div
        style={{
          padding: "8px 20px",
          backgroundColor: "rgba(0, 0, 0, 0.8)",
          display: "flex",
          justifyContent: "center",
          gap: 24,
          fontSize: 12,
          color: "#555",
          flexShrink: 0,
        }}
      >
        <span>← → Scenes</span>
        <span>↑ ↓ Images</span>
        <span>Z Zoom</span>
        <span>K Keeper</span>
        <span>U Unrate</span>
        <span>X Reject</span>
        <span>Esc Close</span>
      </div>
    </div>
  );
}

function SceneCard({
  scene,
  imageMap,
  scoreMap,
  overrides,
  onImageClick,
}: {
  scene: Scene;
  imageMap: Map<number, ImageSummary>;
  scoreMap: Map<number, ImageScore>;
  overrides: OverrideMap;
  onImageClick: (imageId: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const isBurst = scene.image_ids.length > 1;

  const sortedImages = useMemo(() => {
    return getSortedSceneImageIds(scene, scoreMap, overrides)
      .map((id) => ({
        id,
        image: imageMap.get(id),
        score: scoreMap.get(id),
        verdict: getEffectiveVerdict(id, scoreMap, overrides),
      }))
      .filter((item) => item.image != null);
  }, [scene, imageMap, scoreMap, overrides]);

  if (sortedImages.length === 0) return null;

  const coverItem = sortedImages[0];

  if (!isBurst) {
    return (
      <ImageCard
        image={coverItem.image!}
        score={coverItem.score}
        verdict={coverItem.verdict}
        onClick={() => onImageClick(coverItem.id)}
      />
    );
  }

  return (
    <div>
      {!expanded && (
        <div style={{ position: "relative" }}>
          <ImageCard
            image={coverItem.image!}
            score={coverItem.score}
            verdict={coverItem.verdict}
            onClick={() => onImageClick(coverItem.id)}
          />
          <div
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(true);
            }}
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              backgroundColor: "#3b82f6",
              color: "#fff",
              width: 28,
              height: 28,
              borderRadius: 14,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 13,
              fontWeight: "bold",
              boxShadow: "0 2px 6px rgba(0,0,0,0.5)",
              cursor: "pointer",
            }}
          >
            {scene.image_ids.length}
          </div>
        </div>
      )}

      {expanded && (
        <div
          style={{
            backgroundColor: "#111",
            borderRadius: 10,
            border: "1px solid #333",
            padding: 10,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
              padding: "0 4px",
            }}
          >
            <span style={{ fontSize: 13, color: "#999" }}>
              Scene {scene.id} · {scene.image_ids.length} shots
            </span>
            <button
              onClick={() => setExpanded(false)}
              style={{
                background: "none",
                border: "1px solid #555",
                color: "#aaa",
                padding: "4px 12px",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              Collapse
            </button>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: 8,
            }}
          >
            {sortedImages.map((item) => (
              <ImageCard
                key={item.id}
                image={item.image!}
                score={item.score}
                verdict={item.verdict}
                compact
                onClick={() => onImageClick(item.id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ImageCard({
  image,
  score,
  verdict,
  compact = false,
  onClick,
}: {
  image: ImageSummary;
  score?: ImageScore;
  verdict: "keeper" | "reject" | "unrated";
  compact?: boolean;
  onClick?: () => void;
}) {
  const sharpness = score?.sharpness_score;

  return (
    <div
      onClick={onClick}
      style={{
        borderRadius: compact ? 6 : 8,
        overflow: "hidden",
        backgroundColor: "#1a1a1a",
        cursor: onClick ? "pointer" : "default",
        border:
          verdict === "keeper"
            ? "2px solid #22c55e"
            : verdict === "reject"
            ? "2px solid #ef4444"
            : "1px solid #222",
      }}
    >
      <img
        src={convertFileSrc(image.preview_path)}
        alt={image.file_name}
        loading="lazy"
        style={{
          width: "100%",
          display: "block",
          aspectRatio: "3/2",
          objectFit: "cover",
          opacity: verdict === "reject" ? 0.5 : 1,
        }}
      />

      <div style={{ padding: compact ? "6px 8px" : "8px 10px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <p
            style={{
              fontSize: compact ? 11 : 12,
              fontWeight: "bold",
              margin: 0,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              flex: 1,
            }}
          >
            {image.file_name}
          </p>

          {verdict === "keeper" && (
            <span
              style={{
                fontSize: 10,
                fontWeight: "bold",
                color: "#22c55e",
                marginLeft: 8,
                flexShrink: 0,
              }}
            >
              ★ KEEPER
            </span>
          )}
          {verdict === "reject" && (
            <span
              style={{
                fontSize: 10,
                fontWeight: "bold",
                color: "#ef4444",
                marginLeft: 8,
                flexShrink: 0,
              }}
            >
              REJECT
            </span>
          )}
        </div>

        <p
          style={{
            fontSize: compact ? 10 : 11,
            color: "#777",
            margin: "4px 0 0 0",
          }}
        >
          {image.camera_model || "Unknown camera"}
          {image.iso ? ` · ISO ${image.iso}` : ""}
          {image.focal_length_mm ? ` · ${image.focal_length_mm}mm` : ""}
          {compact && sharpness != null && (
            <span style={{ color: "#555" }}>
              {" "}
              · sharpness: {sharpness.toFixed(1)}
            </span>
          )}
        </p>
      </div>
    </div>
  );
}

type FilterMode = "all" | "keepers" | "rejects";

function App() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [folderPath, setFolderPath] = useState("");
  const [manifest, setManifest] = useState<CullManifest | null>(null);
  const [culling, setCulling] = useState(false);
  const [filter, setFilter] = useState<FilterMode>("all");

  // Loupe state
  const [loupeSceneIndex, setLoupeSceneIndex] = useState<number | null>(null);
  const [loupeImageIndex, setLoupeImageIndex] = useState(0);

  // Override state
  const [overrides, setOverrides] = useState<OverrideMap>(new Map());

  // Export state
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);

  // Settings state
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState<CullConfig | null>(null);

  useEffect(() => {
    invoke<CullConfig>("get_config")
      .then((cfg) => setConfig(cfg))
      .catch((err) => console.error("Failed to load config:", err));
  }, []);

  // --- Lookup maps ---
  const imageMap = useMemo(() => {
    const map = new Map<number, ImageSummary>();
    for (const img of images) {
      map.set(img.id, img);
    }
    return map;
  }, [images]);

  const scoreMap = useMemo(() => {
    const map = new Map<number, ImageScore>();
    if (!manifest) return map;
    for (const score of manifest.image_scores) {
      map.set(score.image_id, score);
    }
    return map;
  }, [manifest]);

  // --- Filtered scenes ---
  const filteredScenes = useMemo(() => {
    if (!manifest) return [];
    return manifest.scenes.filter((scene) => {
      if (filter === "all") return true;
      if (filter === "keepers") {
        return getEffectiveKeeperId(scene, overrides) != null;
      }
      if (filter === "rejects") {
        return scene.image_ids.some((id) => {
          return getEffectiveVerdict(id, scoreMap, overrides) === "reject";
        });
      }
      return true;
    });
  }, [manifest, filter, scoreMap, overrides]);

  // --- Override-aware stats ---
  const overrideStats = useMemo(() => {
    if (!manifest) return { keepers: 0, rejects: 0, unrated: 0 };
    let keepers = 0;
    let rejects = 0;
    let unrated = 0;
    for (const score of manifest.image_scores) {
      const v = getEffectiveVerdict(score.image_id, scoreMap, overrides);
      if (v === "keeper") keepers++;
      else if (v === "reject") rejects++;
      else unrated++;
    }
    return { keepers, rejects, unrated };
  }, [manifest, scoreMap, overrides]);

  // --- Loupe: current image data ---
  const loupeData = useMemo(() => {
    if (loupeSceneIndex === null) return null;
    const scene = filteredScenes[loupeSceneIndex];
    if (!scene) return null;
    const sortedIds = getSortedSceneImageIds(scene, scoreMap, overrides);
    const imageId = sortedIds[loupeImageIndex];
    if (imageId === undefined) return null;
    const image = imageMap.get(imageId);
    if (!image) return null;
    return {
      image,
      imageId,
      score: scoreMap.get(imageId),
      scene,
      sceneLabel: `Scene ${loupeSceneIndex + 1} of ${filteredScenes.length}`,
      imageIndexLabel: `Image ${loupeImageIndex + 1} of ${sortedIds.length}`,
      effectiveVerdict: getEffectiveVerdict(imageId, scoreMap, overrides),
    };
  }, [
    loupeSceneIndex,
    loupeImageIndex,
    filteredScenes,
    imageMap,
    scoreMap,
    overrides,
  ]);

  // --- Loupe: open from image click ---
  function handleImageClick(imageId: number) {
    for (let si = 0; si < filteredScenes.length; si++) {
      const scene = filteredScenes[si];
      if (scene.image_ids.includes(imageId)) {
        const sortedIds = getSortedSceneImageIds(scene, scoreMap, overrides);
        const ii = sortedIds.indexOf(imageId);
        setLoupeSceneIndex(si);
        setLoupeImageIndex(ii >= 0 ? ii : 0);
        return;
      }
    }
  }

  // --- Loupe: navigation ---
  const loupeClose = useCallback(() => {
    setLoupeSceneIndex(null);
    setLoupeImageIndex(0);
  }, []);

  const loupePrevScene = useCallback(() => {
    setLoupeSceneIndex((prev) => {
      if (prev === null) return null;
      return prev > 0 ? prev - 1 : filteredScenes.length - 1;
    });
    setLoupeImageIndex(0);
  }, [filteredScenes.length]);

  const loupeNextScene = useCallback(() => {
    setLoupeSceneIndex((prev) => {
      if (prev === null) return null;
      return prev < filteredScenes.length - 1 ? prev + 1 : 0;
    });
    setLoupeImageIndex(0);
  }, [filteredScenes.length]);

  const loupePrevImage = useCallback(() => {
    if (loupeSceneIndex === null) return;
    const scene = filteredScenes[loupeSceneIndex];
    if (!scene) return;
    const total = scene.image_ids.length;
    setLoupeImageIndex((prev) => (prev > 0 ? prev - 1 : total - 1));
  }, [loupeSceneIndex, filteredScenes]);

  const loupeNextImage = useCallback(() => {
    if (loupeSceneIndex === null) return;
    const scene = filteredScenes[loupeSceneIndex];
    if (!scene) return;
    const total = scene.image_ids.length;
    setLoupeImageIndex((prev) => (prev < total - 1 ? prev + 1 : 0));
  }, [loupeSceneIndex, filteredScenes]);

  // --- Override callbacks ---
  const handleMakeKeeper = useCallback(() => {
    if (!loupeData) return;
    setOverrides((prev) => {
      const next = new Map(prev);
      for (const id of loupeData.scene.image_ids) {
        if (next.get(id) === "keeper") {
          next.delete(id);
        }
        if (id === loupeData.scene.keeper_id && id !== loupeData.imageId) {
          next.set(id, "unrated");
        }
      }
      next.set(loupeData.imageId, "keeper");
      return next;
    });
  }, [loupeData]);

  const handleUnrate = useCallback(() => {
    if (!loupeData) return;
    setOverrides((prev) => {
      const next = new Map(prev);
      next.set(loupeData.imageId, "unrated");
      return next;
    });
  }, [loupeData]);

  const handleToggleReject = useCallback(() => {
    if (!loupeData) return;
    setOverrides((prev) => {
      const next = new Map(prev);
      const current = getEffectiveVerdict(
        loupeData.imageId,
        scoreMap,
        prev
      );
      if (current === "reject") {
        next.set(loupeData.imageId, "unrated");
      } else {
        next.set(loupeData.imageId, "reject");
      }
      return next;
    });
  }, [loupeData, scoreMap]);

  // --- Open Folder ---
  async function handleOpenFolder() {
    const selected = await open({ directory: true });
    if (!selected) return;
    setFolderPath(selected as string);
    setLoading(true);
    setImages([]);
    setManifest(null);
    setFilter("all");
    setLoupeSceneIndex(null);
    setOverrides(new Map());
    setExportResult(null);
    try {
      const results = await invoke<ImageSummary[]>("ingest_folder", {
        path: selected,
      });
      setImages(results);
    } catch (err) {
      alert("Error: " + err);
    } finally {
      setLoading(false);
    }
  }

  // --- Cull ---
  async function handleCull() {
    setCulling(true);
    setExportResult(null);
    try {
      const result = await invoke<CullManifest>("cull_images");
      setManifest(result);
      setOverrides(new Map());
    } catch (err) {
      alert("Cull error: " + err);
    } finally {
      setCulling(false);
    }
  }

  // --- Export XMP ---
  async function handleExportXmp() {
    if (!manifest) return;
    setExporting(true);
    setExportResult(null);

    try {
      const verdicts = manifest.image_scores.map((score) => ({
        image_id: score.image_id,
        verdict: getEffectiveVerdict(score.image_id, scoreMap, overrides),
      }));

      const result = await invoke<ExportResponse>("export_xmp", { verdicts });
      setExportResult(result);
    } catch (err) {
      alert("Export error: " + err);
    } finally {
      setExporting(false);
    }
  }

  async function handleSaveSettings(newConfig: CullConfig) {
    try {
      await invoke("save_config", { config: newConfig });
      setConfig(newConfig);
      setShowSettings(false);
    } catch (err) {
      alert("Failed to save settings: " + err);
    }
  }

  return (
    <div
      style={{
        padding: 24,
        fontFamily: "system-ui, sans-serif",
        backgroundColor: "#0a0a0a",
        minHeight: "100vh",
        color: "#ffffff",
      }}
    >
      {/* ====== SETTINGS PANEL ====== */}
      {showSettings && config && (
        <SettingsPanel
          config={config}
          onSave={handleSaveSettings}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* ====== LOUPE OVERLAY ====== */}
      {loupeData && (
        <LoupeView
          image={loupeData.image}
          score={loupeData.score}
          sceneLabel={loupeData.sceneLabel}
          imageIndexLabel={loupeData.imageIndexLabel}
          effectiveVerdict={loupeData.effectiveVerdict}
          onClose={loupeClose}
          onPrevScene={loupePrevScene}
          onNextScene={loupeNextScene}
          onPrevImage={loupePrevImage}
          onNextImage={loupeNextImage}
          onMakeKeeper={handleMakeKeeper}
          onUnrate={handleUnrate}
          onToggleReject={handleToggleReject}
        />
      )}

      {/* ====== HEADER ====== */}
      <div style={{ marginBottom: 24 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <div>
            <h1 style={{ fontSize: 28, margin: 0 }}>keeper.raw</h1>
            <p style={{ color: "#666", margin: "4px 0 16px 0" }}>
              AI-powered photo culling
            </p>
          </div>

          {/* ============================================================
              SETTINGS GEAR BUTTON
              
              Always visible in the top-right corner.
              Opens the Settings panel when clicked.
              ============================================================ */}
          <button
            onClick={() => setShowSettings(true)}
            title="Settings"
            style={{
              background: "none",
              border: "1px solid #333",
              color: "#888",
              padding: "8px 12px",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 16,
            }}
          >
            ⚙ Settings
          </button>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <button
            onClick={handleOpenFolder}
            disabled={loading || culling || exporting}
            style={{
              padding: "10px 24px",
              fontSize: 16,
              cursor: loading || culling || exporting ? "wait" : "pointer",
              backgroundColor: "#ffffff",
              color: "#000000",
              border: "none",
              borderRadius: 6,
              fontWeight: "bold",
            }}
          >
            {loading ? "Extracting previews..." : "Open Folder"}
          </button>

          {images.length > 0 && !manifest && (
            <button
              onClick={handleCull}
              disabled={culling}
              style={{
                padding: "10px 24px",
                fontSize: 16,
                cursor: culling ? "wait" : "pointer",
                backgroundColor: culling ? "#333" : "#22c55e",
                color: "#ffffff",
                border: "none",
                borderRadius: 6,
                fontWeight: "bold",
              }}
            >
              {culling ? "Culling... (this takes a while)" : "Cull Images"}
            </button>
          )}

          {manifest && (
            <button
              onClick={handleExportXmp}
              disabled={exporting}
              style={{
                padding: "10px 24px",
                fontSize: 16,
                cursor: exporting ? "wait" : "pointer",
                backgroundColor: exporting ? "#333" : "#f59e0b",
                color: exporting ? "#888" : "#000000",
                border: "none",
                borderRadius: 6,
                fontWeight: "bold",
              }}
            >
              {exporting ? "Exporting..." : "Export XMP"}
            </button>
          )}
        </div>

        {folderPath && (
          <p style={{ color: "#555", fontSize: 13, marginTop: 8 }}>
            {folderPath}
          </p>
        )}

        {images.length > 0 && !manifest && (
          <p style={{ color: "#888", marginTop: 8 }}>
            {images.length} images loaded — ready to cull
          </p>
        )}

        {manifest && (
          <div
            style={{
              marginTop: 12,
              padding: "12px 16px",
              backgroundColor: "#111",
              borderRadius: 8,
              border: "1px solid #333",
              fontSize: 14,
            }}
          >
            <p style={{ margin: 0, color: "#ccc" }}>
              <strong style={{ color: "#fff" }}>Cull complete</strong> in{" "}
              {manifest.processing_time_secs.toFixed(1)}s
              {overrides.size > 0 && (
                <span style={{ color: "#f59e0b", marginLeft: 8 }}>
                  · {overrides.size} manual override
                  {overrides.size > 1 ? "s" : ""}
                </span>
              )}
            </p>
            <p style={{ margin: "6px 0 0 0", color: "#999" }}>
              {manifest.total_images} images → {manifest.total_scenes}{" "}
              scenes ·{" "}
              <span style={{ color: "#22c55e" }}>
                {overrideStats.keepers} keepers
              </span>{" "}
              ·{" "}
              <span style={{ color: "#ef4444" }}>
                {overrideStats.rejects} rejects
              </span>{" "}
              · {overrideStats.unrated} unrated
            </p>
          </div>
        )}

        {exportResult && (
          <div
            style={{
              marginTop: 8,
              padding: "10px 16px",
              backgroundColor:
                exportResult.errors.length > 0 ? "#1a0a0a" : "#0a1a0a",
              borderRadius: 8,
              border:
                exportResult.errors.length > 0
                  ? "1px solid #ef4444"
                  : "1px solid #22c55e",
              fontSize: 13,
            }}
          >
            <p
              style={{
                margin: 0,
                color:
                  exportResult.errors.length > 0 ? "#ef4444" : "#22c55e",
              }}
            >
              ✓ XMP export complete: {exportResult.files_written} files
              written, {exportResult.files_skipped} skipped
            </p>
            {exportResult.errors.length > 0 && (
              <p
                style={{
                  margin: "4px 0 0 0",
                  color: "#ef4444",
                  fontSize: 12,
                }}
              >
                {exportResult.errors.length} error
                {exportResult.errors.length > 1 ? "s" : ""}:{" "}
                {exportResult.errors[0]}
              </p>
            )}
            <p
              style={{
                margin: "4px 0 0 0",
                color: "#666",
                fontSize: 12,
              }}
            >
              Open the folder in Lightroom, Darktable, or Capture One to see
              your ratings.
            </p>
          </div>
        )}

        {manifest && (
          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            {(["all", "keepers", "rejects"] as FilterMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setFilter(mode)}
                style={{
                  padding: "6px 16px",
                  fontSize: 13,
                  cursor: "pointer",
                  borderRadius: 4,
                  border: "none",
                  fontWeight: filter === mode ? "bold" : "normal",
                  backgroundColor: filter === mode ? "#333" : "transparent",
                  color: filter === mode ? "#fff" : "#777",
                }}
              >
                {mode === "all" && `All (${manifest.total_scenes})`}
                {mode === "keepers" &&
                  `Keepers (${overrideStats.keepers})`}
                {mode === "rejects" &&
                  `Rejects (${overrideStats.rejects})`}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ====== THE GRID ====== */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
          gap: 12,
        }}
      >
        {!manifest &&
          images.map((img) => (
            <div
              key={img.id}
              style={{
                borderRadius: 8,
                overflow: "hidden",
                backgroundColor: "#1a1a1a",
                border: "1px solid #222",
              }}
            >
              <img
                src={convertFileSrc(img.preview_path)}
                alt={img.file_name}
                loading="lazy"
                style={{
                  width: "100%",
                  display: "block",
                  aspectRatio: "3/2",
                  objectFit: "cover",
                }}
              />
              <div style={{ padding: "8px 10px" }}>
                <p
                  style={{
                    fontSize: 12,
                    fontWeight: "bold",
                    margin: 0,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {img.file_name}
                </p>
                <p
                  style={{
                    fontSize: 11,
                    color: "#777",
                    margin: "4px 0 0 0",
                  }}
                >
                  {img.camera_model || "Unknown camera"}
                  {img.iso ? ` · ISO ${img.iso}` : ""}
                  {img.focal_length_mm ? ` · ${img.focal_length_mm}mm` : ""}
                </p>
              </div>
            </div>
          ))}

        {manifest &&
          filteredScenes.map((scene) => (
            <SceneCard
              key={scene.id}
              scene={scene}
              imageMap={imageMap}
              scoreMap={scoreMap}
              overrides={overrides}
              onImageClick={handleImageClick}
            />
          ))}
      </div>

      {/* Empty states */}
      {!loading && images.length === 0 && (
        <div
          style={{ textAlign: "center", padding: "80px 0", color: "#444" }}
        >
          <p style={{ fontSize: 18 }}>No images loaded</p>
          <p style={{ fontSize: 14 }}>
            Click "Open Folder" and select a folder containing RAW files
          </p>
        </div>
      )}

      {manifest && filteredScenes.length === 0 && (
        <div
          style={{ textAlign: "center", padding: "40px 0", color: "#444" }}
        >
          <p style={{ fontSize: 16 }}>
            No {filter === "rejects" ? "rejects" : "keepers"} found
          </p>
          <button
            onClick={() => setFilter("all")}
            style={{
              background: "none",
              border: "1px solid #444",
              color: "#888",
              padding: "6px 16px",
              borderRadius: 4,
              cursor: "pointer",
              marginTop: 8,
            }}
          >
            Show all scenes
          </button>
        </div>
      )}
    </div>
  );
}

export default App;