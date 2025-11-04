import pandas as pd
import numpy as np
import logging
from src.utils.bbox_utils import get_center_of_bbox

# Set up logging
logger = logging.getLogger(__name__)


class PlayerStats:
    def __init__(self):
        """ Initialize PlayerStats instance. """
        pass

    def get_ball_hit_frames(self, ball_positions) -> list:
        """
        Identify frames in which the ball is likely struck/hit by a player.
        This method analyzes a time series of detected ball bounding boxes (one entry
        per video frame) and returns the list of frame indices that correspond to
        likely player hits/shots. The detection pipeline follows these steps:
        1. Input format
            - Expects self.ball_positions to be an iterable with one entry per frame.
              Each entry must support x.get(0, []) and return either an iterable/sequence
              of four numbers [x1, y1, x2, y2] (pixel coordinates of the bounding box)
              or an empty value for missing detections. The returned frame indices map
              directly to the indices of this sequence (DataFrame index).
        2. Preprocessing
            - Converts bboxes to a DataFrame with columns x1,y1,x2,y2.
            - Computes visibility and bbox area.
            - Computes the vertical midpoint (mid_y) of the bbox.
            - Interpolates short gaps (gaps of up to interp_limit frames).
            - Applies a short rolling smoothing to reduce jitter.
        3. Candidate detection
            - Computes frame-to-frame vertical velocity (delta_y) and next-frame delta.
            - Detects candidate hit frames in two ways:
              a) extrema (direction reversals) where delta_y * delta_y_next < 0 and
                  one of the velocity magnitudes exceeds a threshold (min_delta_for_hit).
              b) high acceleration points where accel = delta_y_next - delta_y exceeds
                  min_accel_for_hit.
            - Candidate indices are restricted to frames where both the current and next
              smoothed mid_y exist.
        4. Filtering of candidates
            - Applies multiple heuristics to reduce false positives (all thresholds are
              expressed in pixels or frames unless otherwise noted):
              - Minimum frame separation between accepted hits (min_sep).
              - Reject single-frame teleport-like jumps larger than max_jump_px.
              - Reject extreme acceleration outliers relative to the median motion.
              - Reject if bbox area changes by more than area_change_factor between
                 consecutive frames (likely detection swap).
              - Require a minimum hit acceleration (min_hit_accel).
              - Require minimum outgoing velocity (min_hit_velocity).
              - Accept only hits inside a reasonable vertical play area (min_y_for_hit,
                 max_y_for_hit).
            - These thresholds are currently hard-coded inside the function and can be
              tuned for different video resolutions or detection quality.
        5. Output
            - Returns a list of integer frame indices (DataFrame indices) where hits were
              accepted. If no hits are found, returns an empty list.
            - Side effects: logs summary information about identified hit frames. The
              method does not modify persistent object state (it uses a local DataFrame).
        Args:
             self: The instance providing the attribute `ball_positions` (one entry per
                     frame). Each element must support .get(0, []) and return either an
                     iterable of four numbers [x1, y1, x2, y2] in pixel coordinates or an
                     empty value for missing detection.
        Returns:
             List[int]: Sorted list of frame indices corresponding to detected ball
                            hits/shots (may be empty). Indices correspond to the index of
                            the input `self.ball_positions` sequence.
        Notes:
             - All spatial units (x, y, area, velocity, acceleration) are in pixels or
                pixels/frame and frame counts respectively.
             - The algorithm is heuristic and tuned for typical camera views and ball
                detection noise. Adjust internal thresholds when applying to different
                setups or resolutions.
        """

        # flatten to list of bboxes (empty list for missing)
        ball_positions = [x.get(0, []) for x in ball_positions]
        logger.info(f"{ball_positions}")

        # convert to DataFrame (floats, NaNs for missing)
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"], dtype=float)
        df["ball_hit"] = 0

        # basic visibility/area info (optional later use)
        df["visible"] = df[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
        df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

        # compute mid_y and fill short gaps only
        df["mid_y"] = (df["y1"] + df["y2"]) / 2.0
        interp_limit = 2  # only interpolate gaps <= this many frames
        df["mid_y_filled"] = df["mid_y"].interpolate(
            limit=interp_limit, limit_direction="both"
        )

        # smooth to reduce jitter (small window)
        smooth_window = 2
        df["mid_y_smooth"] = (
            df["mid_y_filled"]
            .rolling(window=smooth_window, min_periods=1, center=False)
            .mean()
        )

        # delta and vectorized extrema detection
        df["delta_y"] = df["mid_y_smooth"].diff()
        df["delta_y_next"] = df["delta_y"].shift(-1)

        # # Debug: log the intermediate DataFrame values
        # logger.info(
        #     "debug frames 1-350:\n"
        #     + df.loc[
        #         1:350,
        #         [
        #             "mid_y",
        #             "mid_y_filled",
        #             "mid_y_smooth",
        #             "delta_y",
        #             "delta_y_next",
        #             "area",
        #             "visible",
        #         ],
        #     ].to_string()
        # )

        min_delta_for_hit = (
            1.0  # Min y-velocity change to be considered a potential hit
        )
        # Relaxed to 1.0 to include small-accel hits in candidates
        min_accel_for_hit = 1.0  # Min acceleration to be considered a potential hit

        # --- Candidate Identification ---
        # A hit can be either a direction reversal (extrema) or a point of high acceleration.

        # 1. Find extrema (peaks and valleys)
        extrema_mask = (df["delta_y"] * df["delta_y_next"] < 0) & (
            (df["delta_y"].abs() >= min_delta_for_hit)
            | (df["delta_y_next"].abs() >= min_delta_for_hit)
        )

        # 2. Find high acceleration points
        df["accel"] = df["delta_y_next"] - df["delta_y"]
        accel_mask = df["accel"].abs() >= min_accel_for_hit

        # Combine candidates and remove duplicates
        candidate_mask = extrema_mask | accel_mask
        valid_pair = df["mid_y_smooth"].notna() & df["mid_y_smooth"].shift(-1).notna()
        candidate_indices = sorted(
            int(i) for i in df.index[candidate_mask & valid_pair]
        )

        # --- Filtering ---
        min_sep = 8  # To prevent double-counting, minimum separation between hits
        max_jump_px = 40.0  # reject single-frame jumps larger than this
        area_change_factor = 2.0  # reject if area changes more than this factor
        min_hit_accel = 7.0 # Minimum acceleration to consider a valid hit
        min_hit_velocity = 15.0 # Minimum outgoing velocity to consider a valid hit
        max_y_for_hit = 450.0 # Reject hits beyond this Y
        min_y_for_hit = 50.0  # Reject hits below this Y
        filtered = []
        last_idx = -999

        # Compute median absolute delta for acceleration outlier rejection
        median_abs_delta = (
            float(np.nanmedian(df["delta_y"].abs().dropna()))
            if not df["delta_y"].dropna().empty
            else 0.0
        )
        for idx in candidate_indices:
            # enforce minimum separation
            if idx - last_idx < min_sep:
                continue

            dy = df["delta_y"].iloc[idx]
            dy_next = df["delta_y_next"].iloc[idx]
            # must be numeric
            if np.isnan(dy) or np.isnan(dy_next):
                continue

            # reject single-frame huge teleport jumps
            if abs(dy) > max_jump_px or abs(dy_next) > max_jump_px:
                logger.debug(
                    f"Reject idx {idx}: large jump dy={dy:.2f}, dy_next={dy_next:.2f}"
                )
                continue

            # reject extreme accel outliers: change in delta is too large relative to median motion
            accel = abs(dy_next - dy)
            if median_abs_delta > 0 and accel > max(20.0, 8.0 * median_abs_delta):
                logger.debug(
                    f"Reject idx {idx}: accel outlier {accel:.2f} >> median_abs_delta {median_abs_delta:.2f}"
                )
                continue

            # reject if bbox area changes massively between i and i+1 (likely detection swap)
            area_i = df["area"].iloc[idx]
            area_next = df["area"].shift(-1).iloc[idx]
            if not np.isnan(area_i) and not np.isnan(area_next) and area_i > 0:
                ratio = (
                    area_next / area_i
                    if area_next >= area_i
                    else area_i / area_next if area_next > 0 else np.inf
                )
                if ratio > area_change_factor:
                    logger.debug(
                        f"Reject idx {idx}: area change ratio {ratio:.2f} (area_i={area_i:.2f}, area_next={area_next:.2f})"
                    )
                    continue

            # Reject if below minimum hit acceleration
            accel = abs(df['delta_y_next'].iloc[idx] - df['delta_y'].iloc[idx])
            if accel < min_hit_accel:
                logger.debug(f"Reject idx {idx}: Accel {accel:.2f} too low for a player hit.")
                continue

            # reject out-of-bounds hits. prevent strange behavior like ball dropping from sky
            if (df['mid_y_smooth'].iloc[idx] > max_y_for_hit) or \
               (df['mid_y_smooth'].iloc[idx] < min_y_for_hit):
                logger.debug(f"Reject idx {idx}: Ball out of reasonable play area (Y={df['mid_y_smooth'].iloc[idx]:.2f}).")
                continue

            # reject low outgoing velocity
            if abs(df['delta_y_next'].iloc[idx]) < min_hit_velocity:
                logger.debug(f"Reject idx {idx}: Low outgoing velocity (V_out={abs(df['delta_y_next'].iloc[idx]):.2f}).")
                continue

            # pass all checks -> accept
            filtered.append(idx)
            last_idx = idx

        # mark filtered hits
        if filtered:
            df.loc[filtered, "ball_hit"] = 1

        frame_nums_with_ball_hits = df.index[df["ball_hit"] == 1].tolist()

        logger.info(
            f"Identified {len(frame_nums_with_ball_hits)} frames with ball hits/shots."
        )
        logger.info(f"Frames with ball hits/shots: {frame_nums_with_ball_hits}")

        return frame_nums_with_ball_hits

    def get_nearest_player_id(self, frames, ball_positions, player_detections, max_distance_px=None):
        """
        If `frames` is an int -> return nearest player_id (or None).
        If `frames` is an iterable of ints -> return dict {frame: player_id_or_None}.
        Uses self.ball_positions and self.player_detections (must be same indexing).
        If max_distance_px provided, returns None when nearest player is farther than that.
        """
        # # handle iterable of frames
        # if isinstance(frames, (list, tuple, set, np.ndarray)):
        #     out = {}
        #     for f in frames:
        #         out[int(f)] = self.get_nearest_player_id(int(f), max_distance_px=max_distance_px)
        #     return out

        # single frame path
        frame_idx = int(frames)
        # bounds checks
        if frame_idx < 0 or frame_idx >= len(ball_positions):
            return None
        if player_detections is None or frame_idx < 0 or frame_idx >= len(player_detections):
            return None

        # extract ball bbox and player detections for the frame (assume formats used elsewhere)
        entry = ball_positions[frame_idx]
        ball_bbox = entry.get(0, []) if hasattr(entry, "get") else entry
        if not ball_bbox or (isinstance(ball_bbox, (list, tuple)) and len(ball_bbox) < 4):
            return None
        try:
            ball_c = get_center_of_bbox(ball_bbox)
        except Exception:
            return None
        if ball_c is None:
            return None

        players_frame = player_detections[frame_idx]
        # build candidate list (id, bbox)
        candidates = []
        if hasattr(players_frame, "items"):
            for pid, bbox in players_frame.items():
                candidates.append((pid, bbox))
        elif isinstance(players_frame, (list, tuple)):
            for i, item in enumerate(players_frame):
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                    candidates.append((item[0], item[1]))
                    continue
                if isinstance(item, dict):
                    pid = item.get("id", item.get("track_id", item.get("player_id", i)))
                    bbox = item.get("bbox") or item.get("box") or item.get("bbox_xyxy")
                    if bbox is None and all(k in item for k in ("x1", "y1", "x2", "y2")):
                        bbox = [item["x1"], item["y1"], item["x2"], item["y2"]]
                    candidates.append((pid, bbox))
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    candidates.append((i, item))
                    continue
        else:
            return None

        best_id = None
        best_dist = float("inf")
        for pid, bbox in candidates:
            if bbox is None or len(bbox) < 4:
                continue
            try:
                pc = get_center_of_bbox(bbox)
            except Exception:
                continue
            if pc is None:
                continue
            dist = float(np.hypot(pc[0] - ball_c[0], pc[1] - ball_c[1]))
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is None:
            return None
        if (max_distance_px is not None) and (best_dist > float(max_distance_px)):
            return None
        return best_id


    def run_player_stats(self,ball_positions, player_detections):
        """ Placeholder for running player stats calculations.
            Args:
                ball_positions: List of ball positions per frame.
                player_detections: List of player detections per frame.
        """

        frames = self.get_ball_hit_frames(ball_positions)
        for frame in frames:
            player_id = self.get_nearest_player_id(frame, ball_positions, player_detections)
            logger.info(f"Frame {frame}: Nearest player ID to ball hit: {player_id}")

        return