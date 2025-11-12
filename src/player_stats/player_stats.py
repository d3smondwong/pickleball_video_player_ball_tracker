from typing import Optional
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import cv2
import logging
from src.utils.bbox_utils import get_center_of_bbox, get_foot_position
from src.mini_court.mini_court import MiniCourt

# Set up logging
logger = logging.getLogger(__name__)


class PlayerStats:
    def __init__(self, cfg: DictConfig):
        """Initialize PlayerStats instance."""
        self.cfg = cfg

    def draw_player_stats(
        self,
        output_video_frames: list,
        ball_hits_df: pd.DataFrame,
        player_metrics: Optional[list] = None,
    ):
        """
        Draw a 4-column player stats panel on each hit frame.
        - output_video_frames: list of frames (BGR numpy arrays)
        - ball_hits_df: DataFrame with at least 'frame_num' and one column per player_id whose value is 1 when that player hit the ball
        - player_metrics: List of dictionaries with player metrics including real-world coordinates.
        Behaviour:
        - Always read player metadata from self.cfg.player_details if present.
        - If config not available, fall back to showing the player_id strings from ball_hits_df columns.
        - Show cumulative "Ball touches" per player and overlay on hit frames.
        """
        if ball_hits_df is None or ball_hits_df.empty:
            return output_video_frames

        # Build id->name mapping from config (preferred). Do NOT use a passed-in player_details.
        id_to_name = {}
        cfg_pd = None
        try:
            cfg_pd = getattr(self.cfg, "player_details", None)
        except Exception:
            cfg_pd = None

        if cfg_pd:
            for key, entry in cfg_pd.items():
                # entry expected like: { height: ..., name: "...", player_id: 1 }
                if hasattr(entry, "get"):
                    pid = entry.get("player_id", None)
                    name = entry.get("name", None)
                else:
                    pid = key
                    name = entry
                if pid is None:
                    try:
                        pid = int(key)
                    except Exception:
                        pid = str(key)
                try:
                    pid_norm = int(pid)
                except Exception:
                    pid_norm = str(pid)
                id_to_name[pid_norm] = str(name) if name is not None else str(pid_norm)

        # Determine player ids to display: prefer config order player_1..player_4, else infer from player_stats
        display_ids = []
        if cfg_pd and any(str(k).startswith("player_") for k in cfg_pd.keys()):
            for key in sorted(cfg_pd.keys()):
                if len(display_ids) >= 4:
                    break
                entry = cfg_pd[key]
                if hasattr(entry, "get"):
                    pid = entry.get("player_id", None)
                    if pid is not None:
                        display_ids.append(pid)
        # fallback: infer up to 4 player ids from ball_hits_df columns (exclude 'frame_num')
        if not display_ids:
            for col in ball_hits_df.columns:
                if col == "frame_num":
                    continue
                if len(display_ids) >= 4:
                    break
                display_ids.append(col)

        # Ensure exactly 4 columns (pad placeholders)
        while len(display_ids) < 4:
            display_ids.append(f"none_{len(display_ids)+1}")

        # Prepare cumulative counts
        counts = {pid: 0 for pid in display_ids}

        # Map hit rows by frame for quick lookup (frame_num -> row Series)
        hit_rows = {}
        for _, row in ball_hits_df.iterrows():
            frm_val = row.get("frame_num", None)
            if frm_val is None:
                continue
            try:
                frm = int(frm_val)
                hit_rows[frm] = row
            except (ValueError, TypeError):
                continue

        # Draw the panel on every frame and update counts when a hit occurs on that frame
        total_frames = len(output_video_frames)

        # --- Data Computation ---
        # Calculate ball touches and distances over time
        ball_touches_by_frame = self.compute_ball_touches_over_time(
            ball_hits_df, display_ids, total_frames, hit_rows
        )

        # Compute player distances over time
        distances_by_frame = self.compute_player_distances_over_time(
            player_metrics, display_ids, total_frames
        )

        panel_w = 420
        panel_h = 120  # Increased panel height for the new row
        for frame_num in range(total_frames):
            frame = output_video_frames[frame_num]

            # Get pre-computed stats for the current frame
            ball_touches = ball_touches_by_frame[frame_num]
            distances = distances_by_frame[frame_num]

            # Draw panel background
            overlay = frame.copy()
            h, w = frame.shape[:2]

            padding = 10
            start_x = padding
            start_y = h - panel_h - padding
            end_x = start_x + panel_w
            end_y = start_y + panel_h
            cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
            alpha = 0.55
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Header row + metric column: draw a left "metric" column, then 4 player columns to the right
            # Layout: left metric column + 4 player columns
            left_col_w = 110
            player_area_w = panel_w - left_col_w
            col_w = player_area_w // 4

            # Draw player names in the top row (to the right of the metric column)
            name_font = cv2.FONT_HERSHEY_SIMPLEX
            name_scale = 0.45
            name_th = 1
            name_y = start_y + 25  # baseline y for the name row
            for i, pid in enumerate(display_ids):
                col_left = start_x + left_col_w + i * col_w
                col_w_inner = col_w
                name = id_to_name.get(pid, None)
                if name is None:
                    name = str(pid)
                (text_w, text_h), text_baseline = cv2.getTextSize(
                    name, name_font, name_scale, name_th
                )
                text_x = int(col_left + (col_w_inner - text_w) / 2)
                # clamp to avoid drawing outside the panel
                text_x = max(
                    start_x + left_col_w + 6,
                    min(text_x, start_x + panel_w - 6 - text_w),
                )
                cv2.putText(
                    frame,
                    name,
                    (text_x, name_y),
                    name_font,
                    name_scale,
                    (255, 255, 255),
                    name_th,
                )

            # Draw vertical separator between metric column and player columns
            sep_x = start_x + left_col_w
            cv2.line(
                frame, (sep_x, start_y + 5), (sep_x, end_y - 5), (120, 120, 120), 1
            )

            # Draw Metrics
            val_font = cv2.FONT_HERSHEY_SIMPLEX
            val_scale = 0.45
            val_th = 1

            # Row 1: Ball Hits
            baseline_y1 = start_y + 60
            metric_name_1 = "Ball Hits"
            m_font = cv2.FONT_HERSHEY_SIMPLEX
            m_scale = 0.45
            m_th = 1
            cv2.putText(
                frame,
                metric_name_1,
                (start_x + 8, baseline_y1),
                m_font,
                m_scale,
                (255, 255, 255),
                m_th,
            )

            for i, pid in enumerate(display_ids):
                col_left = start_x + left_col_w + i * col_w
                col_w_inner = col_w
                val = ball_touches.get(pid, 0)
                text = str(val)
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, val_font, val_scale, val_th
                )
                text_x = int(col_left + (col_w_inner - text_w) / 2)
                text_x = max(
                    col_left + 6, min(text_x, col_left + col_w_inner - 6 - text_w)
                )
                cv2.putText(
                    frame,
                    text,
                    (int(text_x), int(baseline_y1)),
                    val_font,
                    val_scale,
                    (255, 255, 255),
                    val_th,
                )

            # Row 2: Distance Moved
            baseline_y2 = start_y + 100
            metric_name_2 = "Distance (m)"
            cv2.putText(
                frame,
                metric_name_2,
                (start_x + 8, baseline_y2),
                m_font,
                m_scale,
                (255, 255, 255),
                m_th,
            )

            for i, pid in enumerate(display_ids):
                col_left = start_x + left_col_w + i * col_w
                col_w_inner = col_w
                val = distances.get(pid, 0.0)
                text = f"{val:.1f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, val_font, val_scale, val_th
                )
                text_x = int(col_left + (col_w_inner - text_w) / 2)
                text_x = max(
                    col_left + 6, min(text_x, col_left + col_w_inner - 6 - text_w)
                )
                cv2.putText(
                    frame,
                    text,
                    (int(text_x), int(baseline_y2)),
                    val_font,
                    val_scale,
                    (255, 255, 255),
                    val_th,
                )

            # Show current frame number: Development purpose only
            # cv2.putText(
            #     frame,
            #     f"Frame: {frame_num}",
            #     (start_x + 8, start_y + panel_h - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.45,
            #     (180, 180, 180),
            #     1,
            # )

            output_video_frames[frame_num] = frame

        return output_video_frames



    ###
    # Calculate the player movement distance functions
    ###
    def compute_player_distances_over_time(
        self, player_metrics: Optional[list], display_ids: list, total_frames: int
    ):
        """
        Produce a list (len = total_frames) of dicts {player_id: cumulative_distance_meters}
        using real-world coordinates in feet.
        """
        # To store cumulative distances
        distances = {pid: 0.0 for pid in display_ids}

        # To store player's position from previous frame
        last_positions = {pid: None for pid in display_ids}

        # To store distances by frame
        distances_by_frame = []
        FEET_TO_METERS = 0.3048
        MAX_REASONABLE_STEP_FEET_PER_FRAME = 2.0  # Max distance per frame to consider (to filter outliers)

        # Handle case with no player metrics
        if player_metrics is None:
            # If no tracking data, return zeros for all frames
            for _ in range(total_frames):
                distances_by_frame.append(distances.copy())
            return distances_by_frame

        # Iterate through each frame to get the player positions and compute distances using the Euclidean distance
        for f in range(total_frames):
            if f < len(player_metrics):
                frame_metrics = player_metrics[f]
                for pid in display_ids:
                    # Ensure pid from config can be matched with tracker id (int vs str)
                    try:
                        tracker_pid = int(pid)
                    except (ValueError, TypeError):
                        tracker_pid = pid

                    if tracker_pid in frame_metrics:
                        # Use 'feet_pos' for distance calculation
                        current_pos_feet = frame_metrics[tracker_pid].get("feet_pos")

                        if (
                            last_positions[pid] is not None
                            and current_pos_feet is not None
                        ):
                            # Calculate distance in feet using Euclidean distance
                            dist_feet = np.linalg.norm(
                                np.array(current_pos_feet)
                                - np.array(last_positions[pid])
                            )
                            # Filter out impossible jumps
                            if dist_feet < MAX_REASONABLE_STEP_FEET_PER_FRAME:
                                distances[pid] += float(dist_feet)

                        last_positions[pid] = current_pos_feet

            # Convert cumulative distance from feet to meters for display
            distances_in_meters = {
                pid: dist * FEET_TO_METERS for pid, dist in distances.items()
            }
            distances_by_frame.append(distances_in_meters)

        return distances_by_frame

    ###
    # Calculate the ball hits functions
    ###
    def compute_ball_touches_over_time(
        self,
        player_stats: pd.DataFrame,
        display_ids: list,
        total_frames: int,
        hit_rows: dict,
    ):
        """
        Produce a list (len = total_frames) of dicts {player_id: cumulative_count}
        with counts updated on frames that appear in hit_rows.
        """
        counts = {pid: 0 for pid in display_ids}
        counts_by_frame = []
        for f in range(total_frames):
            if f in hit_rows:
                row = hit_rows[f]
                for pid in display_ids:
                    if pid in player_stats.columns:
                        try:
                            if int(row.get(pid, 0)) == 1:
                                counts[pid] = counts.get(pid, 0) + 1
                        except (ValueError, TypeError):
                            if row.get(pid):
                                counts[pid] = counts.get(pid, 0) + 1
            counts_by_frame.append(counts.copy())
        return counts_by_frame

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
        #         1:400,
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
        min_sep = 10  # To prevent double-counting, minimum separation between hits
        max_jump_px = 40.0  # reject single-frame jumps larger than this
        area_change_factor = 2.0  # reject if area changes more than this factor
        min_hit_accel = 4.0  # Minimum acceleration to consider a valid hit
        min_hit_velocity = 6.0  # Minimum outgoing velocity to consider a valid hit
        max_y_for_hit = 450.0  # Reject hits beyond this Y
        min_y_for_hit = 50.0  # Reject hits below this Y
        filtered = []
        last_idx = -999

        skip_initial = int(getattr(self.cfg, "skip_initial_frames", 15))
        if skip_initial > 0:
            candidate_indices = [i for i in candidate_indices if i >= skip_initial]

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
            accel = abs(df["delta_y_next"].iloc[idx] - df["delta_y"].iloc[idx])
            if accel < min_hit_accel:
                logger.debug(
                    f"Reject idx {idx}: Accel {accel:.2f} too low for a player hit."
                )
                continue

            # reject out-of-bounds hits. prevent strange behavior like ball dropping from sky
            if (df["mid_y_smooth"].iloc[idx] > max_y_for_hit) or (
                df["mid_y_smooth"].iloc[idx] < min_y_for_hit
            ):
                logger.debug(
                    f"Reject idx {idx}: Ball out of reasonable play area (Y={df['mid_y_smooth'].iloc[idx]:.2f})."
                )
                continue

            # reject low outgoing velocity
            if abs(df["delta_y_next"].iloc[idx]) < min_hit_velocity:
                logger.debug(
                    f"Reject idx {idx}: Low outgoing velocity (V_out={abs(df['delta_y_next'].iloc[idx]):.2f})."
                )
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

    def get_nearest_player_id(
        self, frames, ball_positions, player_detections, max_distance_px=None
    ):
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
        if (
            player_detections is None
            or frame_idx < 0
            or frame_idx >= len(player_detections)
        ):
            return None

        # extract ball bbox and player detections for the frame
        entry = ball_positions[frame_idx]
        ball_bbox = entry.get(0, []) if hasattr(entry, "get") else entry
        if not ball_bbox or (
            isinstance(ball_bbox, (list, tuple)) and len(ball_bbox) < 4
        ):
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
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[1], (list, tuple))
                ):
                    candidates.append((item[0], item[1]))
                    continue
                if isinstance(item, dict):
                    pid = item.get("id", item.get("track_id", item.get("player_id", i)))
                    bbox = item.get("bbox") or item.get("box") or item.get("bbox_xyxy")
                    if bbox is None and all(
                        k in item for k in ("x1", "y1", "x2", "y2")
                    ):
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

    def run_ball_hits_df(
        self, ball_positions, player_detections, max_distance_px: None
    ):
        """
        Build player_stats DataFrame using only player_ids that were nearest at hit frames.
        Index rows per hit frame ('frame_num'), columns are the detected nearest player_ids,
        cell value = 1 if that player was nearest at that hit, else 0.
        """
        # find hit frames
        frames = self.get_ball_hit_frames(ball_positions)

        # find nearest player for each hit frame
        nearest_per_frame = []
        for f in frames:
            pid = self.get_nearest_player_id(f, ball_positions, player_detections)
            nearest_per_frame.append((int(f), pid))

        # build ordered unique list of player_ids that actually were nearest on a hit
        unique_ids = []
        for _, pid in nearest_per_frame:
            if pid is None:
                continue
            if pid not in unique_ids:
                unique_ids.append(pid)

        # build rows: only include columns for players that were actually nearest
        rows = []
        for f, pid in nearest_per_frame:
            row = {"frame_num": int(f)}
            for uid in unique_ids:
                row[uid] = 0
            if pid is not None and pid in unique_ids:
                row[pid] = 1
            rows.append(row)

        # create DataFrame with columns: frame_num then player ids in discovery order
        cols = ["frame_num"] + unique_ids
        if rows:
            ball_hits_df = pd.DataFrame(rows)
            # ensure columns exist and order enforced
            for c in cols:
                if c not in ball_hits_df.columns:
                    ball_hits_df[c] = 0
            ball_hits_df = ball_hits_df[cols]
        else:
            ball_hits_df = pd.DataFrame(columns=cols)

        self.ball_hits_df = ball_hits_df
        logger.info(f"\n {ball_hits_df}")

        return ball_hits_df
