#!/usr/bin/env python3
"""Convert generic nuScenes info PKLs into GaussianAD-compatible PKLs.

This script converts
- data/nuscenes_cam/nuscenes_infos_train.pkl
- data/nuscenes_cam/nuscenes_infos_val.pkl
into the format expected by this repository:
- infos: Dict[scene_token, List[frame_info]]
- metadata: List[(scene_token, frame_idx)]

It also rebuilds `info['data']` entries (LIDAR_TOP + 6 cameras) from
nuScenes devkit, and fills missing fields required by dataset loading.
"""

import argparse
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mmengine
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box
from shapely.strtree import STRtree
from tqdm import tqdm


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
SENSORS = ["LIDAR_TOP"] + CAMERAS
FUTURE_STEPS = 6
HISTORY_STEPS = 2
FUTURE_STEP_SECONDS = 0.5
DEFAULT_MAP_PC_RANGE = (-30.0, -30.0, -2.0, 30.0, 30.0, 2.0)
DEFAULT_EGO_FUT_CMD = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DEFAULT_MIN_MAP_LINE_LENGTH = 0.5
DEFAULT_MAP_SIMPLIFY_TOLERANCE = 0.05
STATIC_COMMAND_DISP_THRESHOLD = 1.0
STATIC_COMMAND_PATH_THRESHOLD = 2.0
TURN_LATERAL_THRESHOLD = 1.0
TURN_YAW_THRESHOLD = 0.20
TURN_YAW_SCORE_WEIGHT = 4.0
CMD_RIGHT_IDX = 0
CMD_STRAIGHT_IDX = 1
CMD_LEFT_IDX = 2
PLANNER_TYPE_BY_NAME = {
    "pedestrian": 2,
    "car": 14,
    "truck": 14,
    "construction_vehicle": 14,
    "bus": 14,
    "trailer": 14,
    "motorcycle": 14,
    "bicycle": 14,
}
NUSCENE_LOCATIONS = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)
MAP_LINE_CLASSES = ("road_divider", "lane_divider")
MAP_PED_CLASSES = ("ped_crossing",)
MAP_POLYGON_CLASSES = ("road_segment", "lane")


def _resolve_surroundocc_path(lidar_filename: str, occ_dir: Optional[str]) -> Optional[str]:
    if not occ_dir:
        return None

    occ_file = os.path.join(occ_dir, os.path.basename(lidar_filename) + ".npy")
    if os.path.exists(occ_file):
        return occ_file
    return None


def _as_numpy(x: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _zeros(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _empty_gt_map() -> Dict[str, List]:
    return {"divider": [], "ped_crossing": [], "boundary": []}


def _make_transform(rotation: Iterable[float], translation: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    transform[:3, 3] = _as_numpy(translation, np.float32)
    return transform


def _compose_quaternion(a: Iterable[float], b: Iterable[float]) -> Quaternion:
    return Quaternion(a) * Quaternion(b)


def _normalize_ego_fut_cmd(cmd: Any) -> np.ndarray:
    arr = _as_numpy(cmd, np.float32).reshape(-1)
    if arr.size == 3:
        if np.isclose(arr.sum(), 0.0):
            return DEFAULT_EGO_FUT_CMD.copy()
        out = np.zeros(3, dtype=np.float32)
        out[int(np.argmax(arr))] = 1.0
        return out
    if arr.size == 1 and np.isfinite(arr[0]):
        out = np.zeros(3, dtype=np.float32)
        out[int(np.clip(arr[0], 0, 2))] = 1.0
        return out
    return DEFAULT_EGO_FUT_CMD.copy()


def _infer_ego_fut_cmd(step_trajs: np.ndarray, yaw_delta: float = 0.0) -> np.ndarray:
    if step_trajs.size == 0:
        return DEFAULT_EGO_FUT_CMD.copy()

    cumulative = np.cumsum(step_trajs, axis=0)
    final_disp = float(np.linalg.norm(cumulative[-1]))
    path_len = float(np.linalg.norm(step_trajs, axis=1).sum())
    if final_disp < STATIC_COMMAND_DISP_THRESHOLD or path_len < STATIC_COMMAND_PATH_THRESHOLD:
        return DEFAULT_EGO_FUT_CMD.copy()

    final_lateral = float(cumulative[-1, 0])
    turn_score = final_lateral - TURN_YAW_SCORE_WEIGHT * float(yaw_delta)
    out = np.zeros(3, dtype=np.float32)
    if turn_score > TURN_LATERAL_THRESHOLD or yaw_delta < -TURN_YAW_THRESHOLD:
        out[CMD_RIGHT_IDX] = 1.0
    elif turn_score < -TURN_LATERAL_THRESHOLD or yaw_delta > TURN_YAW_THRESHOLD:
        out[CMD_LEFT_IDX] = 1.0
    else:
        out[CMD_STRAIGHT_IDX] = 1.0
    return out


def _build_agent_lcf_feat(
    gt_boxes: np.ndarray,
    gt_velocity: np.ndarray,
    gt_names: np.ndarray,
    existing_feat: Any,
) -> np.ndarray:
    n = int(gt_boxes.shape[0])
    out = _zeros((n, 9), np.float32)

    arr = _as_numpy(existing_feat, np.float32)
    if arr.size == n * 9:
        out = arr.reshape(n, 9).copy()
    elif arr.ndim == 2 and arr.shape[0] == n:
        width = min(arr.shape[1], 9)
        out[:, :width] = arr[:, :width]

    if n == 0:
        return out

    out[:, 0:2] = gt_boxes[:, 0:2]
    if gt_boxes.shape[1] > 6:
        out[:, 2] = gt_boxes[:, 6]
    out[:, 3:5] = gt_velocity
    if gt_boxes.shape[1] > 5:
        out[:, 5:8] = gt_boxes[:, 3:6]
    out[:, 8] = np.asarray([PLANNER_TYPE_BY_NAME.get(str(name), 0) for name in gt_names], dtype=np.float32)
    return out


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _map_category_name(category_name: str) -> Optional[str]:
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    return None


def _coords_from_line(instance: LineString) -> Optional[List[List[float]]]:
    coords = np.asarray(instance.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None
    return coords[:, :2].tolist()


def _clean_local_line(line: LineString, min_line_length: float, simplify_tolerance: float = 0.0) -> Optional[LineString]:
    if line.is_empty:
        return None
    if simplify_tolerance > 0 and line.length > simplify_tolerance:
        line = line.simplify(simplify_tolerance, preserve_topology=True)
    coords = np.asarray(line.coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None
    coords = coords[:, :2]
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.shape[0] < 2:
        return None

    keep = [0]
    for idx in range(1, coords.shape[0]):
        if np.linalg.norm(coords[idx] - coords[keep[-1]]) > 1e-3:
            keep.append(idx)
    coords = coords[keep]
    if coords.shape[0] < 2:
        return None

    cleaned = LineString(coords)
    if cleaned.is_empty or cleaned.length < min_line_length:
        return None
    return cleaned


def _sanitize_gt_map(
    gt_map: Dict[str, List[List[List[float]]]],
    pc_range: Tuple[float, float, float, float, float, float],
    min_line_length: float,
    simplify_tolerance: float,
) -> Dict[str, List[List[List[float]]]]:
    local_patch = box(float(pc_range[0]), float(pc_range[1]), float(pc_range[3]), float(pc_range[4]))
    sanitized = _empty_gt_map()
    for class_name in sanitized.keys():
        for coords in gt_map.get(class_name, []):
            arr = np.asarray(coords, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                continue
            arr = arr[:, :2]
            if not np.isfinite(arr).all():
                arr = arr[np.isfinite(arr).all(axis=1)]
            if arr.shape[0] < 2:
                continue
            line = LineString(arr)
            if line.is_empty:
                continue
            clipped = line.intersection(local_patch)
            for clipped_line in _iter_lines(clipped):
                cleaned = _clean_local_line(clipped_line, min_line_length, simplify_tolerance)
                if cleaned is None:
                    continue
                out_coords = _coords_from_line(cleaned)
                if out_coords is not None:
                    sanitized[class_name].append(out_coords)
    return sanitized


def _inside_xy_range(points: np.ndarray, pc_range: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    return (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] <= pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] <= pc_range[4])
    )


def _apply_plan_quality_mask(
    step_offsets: np.ndarray,
    fut_masks: np.ndarray,
    full_future: bool,
    pc_range: Tuple[float, float, float, float, float, float],
    zero_incomplete: bool,
    mask_outside_range: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    step_offsets = step_offsets.copy()
    fut_masks = fut_masks.copy()

    if zero_incomplete and not full_future:
        step_offsets[:] = 0.0
        fut_masks[:] = 0.0
        return step_offsets, fut_masks

    if mask_outside_range and np.any(fut_masks > 0):
        cumulative = np.cumsum(step_offsets, axis=0)
        inside = _inside_xy_range(cumulative, pc_range)
        inside_prefix = np.cumprod(inside.astype(np.float32))
        fut_masks *= inside_prefix
        step_offsets[fut_masks <= 0] = 0.0

    return step_offsets, fut_masks


def _lidar2global_from_info(info: Dict[str, Any]) -> np.ndarray:
    return _make_transform(info["ego2global_rotation"], info["ego2global_translation"]) @ _make_transform(
        info["lidar2ego_rotation"], info["lidar2ego_translation"]
    )


def _lidar_yaw_from_info(info: Dict[str, Any]) -> float:
    return float(quaternion_yaw(_compose_quaternion(info["ego2global_rotation"], info["lidar2ego_rotation"])))


def _iter_lines(geom) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "LinearRing":
        yield LineString(geom)
    elif geom.geom_type == "MultiLineString":
        for sub_geom in geom.geoms:
            yield from _iter_lines(sub_geom)
    elif geom.geom_type == "GeometryCollection":
        for sub_geom in geom.geoms:
            yield from _iter_lines(sub_geom)


def _iter_polygons(geom) -> Iterable:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for sub_geom in geom.geoms:
            yield sub_geom
    elif geom.geom_type == "GeometryCollection":
        for sub_geom in geom.geoms:
            yield from _iter_polygons(sub_geom)


class _MapAnnotationBuilder:
    def __init__(
        self,
        dataroot: str,
        patch_size: Tuple[float, float],
        min_line_length: float = DEFAULT_MIN_MAP_LINE_LENGTH,
        simplify_tolerance: float = DEFAULT_MAP_SIMPLIFY_TOLERANCE,
    ) -> None:
        self.patch_size = patch_size
        self.min_line_length = float(min_line_length)
        self.simplify_tolerance = float(simplify_tolerance)
        self.nusc_maps: Dict[str, NuScenesMap] = {}
        self.explorers: Dict[str, NuScenesMapExplorer] = {}
        self.cached_geometries: Dict[str, Dict[str, List]] = defaultdict(dict)
        self.cached_trees: Dict[str, Dict[str, Optional[STRtree]]] = defaultdict(dict)
        for location in NUSCENE_LOCATIONS:
            self.nusc_maps[location] = NuScenesMap(dataroot=dataroot, map_name=location)
            self.explorers[location] = NuScenesMapExplorer(self.nusc_maps[location])
            self._build_location_cache(location)

    def _cache_layer(self, location: str, layer_name: str, geometries: List) -> None:
        self.cached_geometries[location][layer_name] = geometries
        self.cached_trees[location][layer_name] = STRtree(geometries) if geometries else None

    def _build_location_cache(self, location: str) -> None:
        explorer = self.explorers[location]

        for layer_name in MAP_LINE_CLASSES:
            geometries = []
            for record in getattr(explorer.map_api, layer_name):
                line = explorer.map_api.extract_line(record["line_token"])
                if not line.is_empty:
                    geometries.append(line)
            self._cache_layer(location, layer_name, geometries)

        for layer_name in MAP_POLYGON_CLASSES:
            geometries = []
            for record in getattr(explorer.map_api, layer_name):
                polygon = explorer.map_api.extract_polygon(record["polygon_token"])
                if polygon.is_valid and not polygon.is_empty:
                    geometries.append(polygon)
            self._cache_layer(location, layer_name, geometries)

        ped_geometries = []
        for record in getattr(explorer.map_api, "ped_crossing"):
            polygon = explorer.map_api.extract_polygon(record["polygon_token"])
            if polygon.is_valid and not polygon.is_empty:
                ped_geometries.append(polygon)
        self._cache_layer(location, "ped_crossing", ped_geometries)

    def _query_candidates(self, location: str, layer_name: str, patch) -> List:
        tree = self.cached_trees[location].get(layer_name)
        if tree is None:
            return []
        return list(tree.query(patch))

    def _get_map_geom(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        layer_names: Iterable[str],
        location: str,
    ) -> List[Tuple[str, List]]:
        map_geom = []
        for layer_name in layer_names:
            if layer_name in MAP_LINE_CLASSES:
                geoms = self._get_divider_line(patch_box, patch_angle, layer_name, location)
            elif layer_name in MAP_POLYGON_CLASSES:
                geoms = self._get_contour_line(patch_box, patch_angle, layer_name, location)
            elif layer_name in MAP_PED_CLASSES:
                geoms = self._get_ped_crossing_line(patch_box, patch_angle, location)
            else:
                continue
            map_geom.append((layer_name, geoms))
        return map_geom

    def _get_divider_line(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        layer_name: str,
        location: str,
    ) -> List:
        explorer = self.explorers[location]
        if layer_name not in explorer.map_api.non_geometric_line_layers:
            raise ValueError(f"{layer_name} is not a line layer")

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        for line in self._query_candidates(location, layer_name, patch):
            if line.is_empty:
                continue
            new_line = line.intersection(patch)
            if new_line.is_empty:
                continue
            new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            line_list.append(new_line)
        return line_list

    def _get_contour_line(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        layer_name: str,
        location: str,
    ) -> List:
        explorer = self.explorers[location]
        if layer_name not in explorer.map_api.non_geometric_polygon_layers:
            raise ValueError(f"{layer_name} is not a polygonal layer")

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = explorer.get_patch_coord(patch_box, patch_angle)

        polygon_list = []
        for polygon in self._query_candidates(location, layer_name, patch):
            if not polygon.is_valid:
                continue
            new_polygon = polygon.intersection(patch)
            if new_polygon.is_empty:
                continue
            new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            if new_polygon.geom_type == "Polygon":
                new_polygon = MultiPolygon([new_polygon])
            polygon_list.append(new_polygon)
        return polygon_list

    def _get_ped_crossing_line(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        location: str,
    ) -> List:
        explorer = self.explorers[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = explorer.get_patch_coord(patch_box, patch_angle)

        polygon_list = []
        for polygon in self._query_candidates(location, "ped_crossing", patch):
            if not polygon.is_valid:
                continue
            new_polygon = polygon.intersection(patch)
            if new_polygon.is_empty:
                continue
            new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            if new_polygon.geom_type == "Polygon":
                new_polygon = MultiPolygon([new_polygon])
            polygon_list.append(new_polygon)
        return polygon_list

    def _clean_line(self, line: LineString) -> Optional[LineString]:
        if line.is_empty:
            return None
        if self.simplify_tolerance > 0 and line.length > self.simplify_tolerance:
            line = line.simplify(self.simplify_tolerance, preserve_topology=True)
        coords = np.asarray(line.coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[0] < 2:
            return None

        keep = [0]
        for idx in range(1, coords.shape[0]):
            if np.linalg.norm(coords[idx, :2] - coords[keep[-1], :2]) > 1e-3:
                keep.append(idx)
        coords = coords[keep, :2]
        if coords.shape[0] < 2:
            return None

        cleaned = LineString(coords)
        if cleaned.is_empty or cleaned.length < self.min_line_length:
            return None
        return cleaned

    def _one_type_line_geom_to_instances(self, line_geom: Iterable) -> List[LineString]:
        line_instances: List[LineString] = []
        for geom in line_geom:
            for line in _iter_lines(geom):
                cleaned = self._clean_line(line)
                if cleaned is not None:
                    line_instances.append(cleaned)
        return line_instances

    def _line_geoms_to_instances(self, line_geom: Iterable[Tuple[str, List]]) -> Dict[str, List[LineString]]:
        line_instances_dict: Dict[str, List[LineString]] = {}
        for line_type, a_type_of_lines in line_geom:
            line_instances_dict[line_type] = self._one_type_line_geom_to_instances(a_type_of_lines)
        return line_instances_dict

    def _ped_poly_geoms_to_instances(self, ped_geom: Iterable[Tuple[str, List]]) -> List[LineString]:
        ped = ped_geom[0][1]
        if not ped:
            return []
        union_segments = ops.unary_union(ped)
        if union_segments.is_empty:
            return []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        for poly in _iter_polygons(union_segments):
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def _poly_geoms_to_instances(self, polygon_geom: Iterable[Tuple[str, List]]) -> List[LineString]:
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        if not roads and not lanes:
            return []
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        if union_segments.is_empty:
            return []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        for poly in _iter_polygons(union_segments):
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def build_for_info(self, info: Dict[str, Any]) -> Dict[str, List[List[List[float]]]]:
        lidar2global = _lidar2global_from_info(info)
        lidar_translation = lidar2global[:3, 3]
        lidar_rotation = _compose_quaternion(info["ego2global_rotation"], info["lidar2ego_rotation"])
        patch_box = (float(lidar_translation[0]), float(lidar_translation[1]), self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(lidar_rotation) / np.pi * 180.0
        location = info["map_location"]

        gt_map = _empty_gt_map()

        line_geom = self._get_map_geom(patch_box, patch_angle, MAP_LINE_CLASSES, location)
        for instances in self._line_geoms_to_instances(line_geom).values():
            for instance in instances:
                coords = _coords_from_line(instance)
                if coords is not None:
                    gt_map["divider"].append(coords)

        ped_geom = self._get_map_geom(patch_box, patch_angle, MAP_PED_CLASSES, location)
        for instance in self._ped_poly_geoms_to_instances(ped_geom):
            coords = _coords_from_line(instance)
            if coords is not None:
                gt_map["ped_crossing"].append(coords)

        polygon_geom = self._get_map_geom(patch_box, patch_angle, MAP_POLYGON_CLASSES, location)
        for instance in self._poly_geoms_to_instances(polygon_geom):
            coords = _coords_from_line(instance)
            if coords is not None:
                gt_map["boundary"].append(coords)

        return gt_map


def _annotation_to_reference_lidar(
    ann: Dict[str, Any],
    ref_lidar_calib: Dict[str, Any],
    ref_lidar_pose: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
    box.translate(-_as_numpy(ref_lidar_pose["translation"], np.float32))
    box.rotate(Quaternion(ref_lidar_pose["rotation"]).inverse)
    box.translate(-_as_numpy(ref_lidar_calib["translation"], np.float32))
    box.rotate(Quaternion(ref_lidar_calib["rotation"]).inverse)
    center = np.asarray(box.center, dtype=np.float32)
    size = np.asarray(box.wlh, dtype=np.float32)
    yaw = float(quaternion_yaw(box.orientation))
    return center, size, yaw


def _match_gt_boxes_to_annotations(
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    candidates: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    matches: Dict[int, Dict[str, Any]] = {}
    used = set()
    for gt_idx in range(int(gt_boxes.shape[0])):
        gt_name = str(gt_names[gt_idx])
        gt_center = gt_boxes[gt_idx, :3]
        gt_size = np.sort(gt_boxes[gt_idx, 3:6])
        best_j = None
        best_cost = float("inf")
        best_center_dist = float("inf")
        for cand_idx, cand in enumerate(candidates):
            if cand_idx in used or cand["name"] != gt_name:
                continue
            center_dist = float(np.linalg.norm(gt_center - cand["center"]))
            size_dist = float(np.abs(gt_size - np.sort(cand["size"])).sum())
            cost = center_dist + 0.25 * size_dist
            if cost < best_cost:
                best_cost = cost
                best_center_dist = center_dist
                best_j = cand_idx
        if best_j is None:
            continue
        if best_center_dist > 2.5 or best_cost > 4.0:
            continue
        matches[gt_idx] = candidates[best_j]
        used.add(best_j)
    return matches


def _fill_scene_agent_future_labels(scene_infos: List[Dict[str, Any]], nusc: NuScenes) -> None:
    for info in scene_infos:
        gt_boxes = _as_numpy(info.get("gt_boxes", _zeros((0, 7), np.float32)), np.float32)
        gt_names = np.asarray(info.get("gt_names", np.array([], dtype=object)))
        gt_velocity = _as_numpy(info.get("gt_velocity", _zeros((gt_boxes.shape[0], 2), np.float32)), np.float32)
        num_agents = int(gt_boxes.shape[0])

        gt_agent_fut_trajs = _zeros((num_agents, FUTURE_STEPS * 2), np.float32)
        gt_agent_fut_masks = _zeros((num_agents, FUTURE_STEPS), np.float32)
        gt_agent_fut_goal = _zeros((num_agents,), np.float32)
        gt_agent_fut_yaw = _zeros((num_agents, FUTURE_STEPS), np.float32)

        if num_agents == 0:
            info["gt_agent_fut_trajs"] = gt_agent_fut_trajs
            info["gt_agent_fut_masks"] = gt_agent_fut_masks
            info["gt_agent_fut_goal"] = gt_agent_fut_goal
            info["gt_agent_fut_yaw"] = gt_agent_fut_yaw
            info["gt_agent_lcf_feat"] = _build_agent_lcf_feat(gt_boxes, gt_velocity, gt_names, info.get("gt_agent_lcf_feat"))
            continue

        sample = nusc.get("sample", info["token"])
        ref_lidar_calib = info["data"]["LIDAR_TOP"]["calib"]
        ref_lidar_pose = info["data"]["LIDAR_TOP"]["pose"]

        candidates = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            mapped_name = _map_category_name(ann["category_name"])
            if mapped_name is None:
                continue
            center, size, yaw = _annotation_to_reference_lidar(ann, ref_lidar_calib, ref_lidar_pose)
            candidates.append(
                {
                    "ann": ann,
                    "name": mapped_name,
                    "center": center,
                    "size": size,
                    "yaw": yaw,
                }
            )

        matches = _match_gt_boxes_to_annotations(gt_boxes, gt_names, candidates)
        for gt_idx, matched in matches.items():
            current_ann = matched["ann"]
            prev_center = gt_boxes[gt_idx, :2].astype(np.float32)
            prev_yaw = float(gt_boxes[gt_idx, 6]) if gt_boxes.shape[1] > 6 else float(matched["yaw"])
            last_future_center = None

            for step in range(FUTURE_STEPS):
                next_ann_token = current_ann.get("next")
                if not next_ann_token:
                    break
                next_ann = nusc.get("sample_annotation", next_ann_token)
                future_center, _, future_yaw = _annotation_to_reference_lidar(next_ann, ref_lidar_calib, ref_lidar_pose)
                future_xy = future_center[:2]

                gt_agent_fut_trajs[gt_idx, step * 2:(step + 1) * 2] = future_xy - prev_center
                gt_agent_fut_masks[gt_idx, step] = 1.0
                gt_agent_fut_yaw[gt_idx, step] = -_wrap_to_pi(future_yaw - prev_yaw)

                prev_center = future_xy
                prev_yaw = future_yaw
                current_ann = next_ann
                last_future_center = future_xy

            if last_future_center is not None:
                gt_agent_fut_goal[gt_idx] = float(np.linalg.norm(last_future_center - gt_boxes[gt_idx, :2]))

        info["gt_agent_fut_trajs"] = gt_agent_fut_trajs
        info["gt_agent_fut_masks"] = gt_agent_fut_masks
        info["gt_agent_fut_goal"] = gt_agent_fut_goal
        info["gt_agent_fut_yaw"] = gt_agent_fut_yaw
        info["gt_agent_lcf_feat"] = _build_agent_lcf_feat(gt_boxes, gt_velocity, gt_names, info.get("gt_agent_lcf_feat"))


def _fill_scene_planner_labels(
    scene_infos: List[Dict[str, Any]],
    plan_valid_pc_range: Tuple[float, float, float, float, float, float],
    zero_incomplete_plan: bool,
    mask_plan_outside_range: bool,
) -> None:
    for idx, info in enumerate(scene_infos):
        lidar2global = _lidar2global_from_info(info)
        global2lidar = np.linalg.inv(lidar2global)
        current_yaw = _lidar_yaw_from_info(info)
        current_ego = np.concatenate([
            _as_numpy(info["ego2global_translation"], np.float32),
            np.array([1.0], dtype=np.float32),
        ])
        current_ego_in_lidar = global2lidar @ current_ego

        future_positions = []
        future_yaws = []
        for step in range(1, FUTURE_STEPS + 1):
            if idx + step >= len(scene_infos):
                break
            future_info = scene_infos[idx + step]
            future_ego = np.concatenate([
                _as_numpy(future_info["ego2global_translation"], np.float32),
                np.array([1.0], dtype=np.float32),
            ])
            future_in_lidar = global2lidar @ future_ego
            future_positions.append(future_in_lidar[:2] - current_ego_in_lidar[:2])
            future_yaws.append(_lidar_yaw_from_info(future_info))

        step_trajs = _zeros((FUTURE_STEPS, 2), np.float32)
        fut_masks = _zeros((FUTURE_STEPS,), np.float32)
        ego_lcf_feat = _zeros((9,), np.float32)
        if future_positions:
            cumulative = np.asarray(future_positions, dtype=np.float32)
            step_offsets = cumulative.copy()
            if cumulative.shape[0] > 1:
                step_offsets[1:] = cumulative[1:] - cumulative[:-1]
            step_trajs[: cumulative.shape[0]] = step_offsets
            fut_masks[: cumulative.shape[0]] = 1.0
            full_future = len(future_positions) == FUTURE_STEPS
            step_trajs, fut_masks = _apply_plan_quality_mask(
                step_trajs,
                fut_masks,
                full_future=full_future,
                pc_range=plan_valid_pc_range,
                zero_incomplete=zero_incomplete_plan,
                mask_outside_range=mask_plan_outside_range,
            )
            yaw_delta = _wrap_to_pi(future_yaws[-1] - current_yaw) if future_yaws else 0.0
            valid_steps = step_trajs[fut_masks > 0]
            info["gt_ego_fut_cmd"] = _infer_ego_fut_cmd(valid_steps, yaw_delta=yaw_delta)

            first_dt = max(
                (float(scene_infos[idx + 1]["timestamp"]) - float(info["timestamp"])) / 1e6,
                1e-3,
            ) if idx + 1 < len(scene_infos) else FUTURE_STEP_SECONDS
            ego_lcf_feat[0:2] = step_trajs[0] / first_dt if fut_masks[0] > 0 else 0.0
            ego_lcf_feat[7] = float(np.linalg.norm(ego_lcf_feat[0:2]))
            ego_lcf_feat[8] = float(yaw_delta / max(len(future_yaws) * FUTURE_STEP_SECONDS, 1e-3))
        else:
            info["gt_ego_fut_cmd"] = DEFAULT_EGO_FUT_CMD.copy()

        info["gt_ego_fut_trajs"] = step_trajs
        info["gt_ego_fut_masks"] = fut_masks
        info["gt_ego_lcf_feat"] = ego_lcf_feat
        info["fut_valid_flag"] = np.array(bool(np.count_nonzero(fut_masks) == FUTURE_STEPS), dtype=np.bool_)

        history = _zeros((HISTORY_STEPS, 2), np.float32)
        history_positions = []
        for step in range(HISTORY_STEPS, 0, -1):
            prev_idx = idx - step
            if prev_idx < 0:
                continue
            prev_info = scene_infos[prev_idx]
            prev_ego = np.concatenate([
                _as_numpy(prev_info["ego2global_translation"], np.float32),
                np.array([1.0], dtype=np.float32),
            ])
            prev_in_lidar = global2lidar @ prev_ego
            history_positions.append(prev_in_lidar[:2] - current_ego_in_lidar[:2])
        if history_positions:
            history[-len(history_positions):] = np.asarray(history_positions, dtype=np.float32)
        info["gt_ego_his_trajs"] = history


def _get_token(info: Dict[str, Any]) -> Optional[str]:
    for k in ("token", "sample_token", "sample_idx"):
        if k in info and info[k] is not None:
            return str(info[k])
    return None


def _get_list_infos(raw_infos: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_infos, list):
        return raw_infos
    if isinstance(raw_infos, dict):
        merged = []
        for v in raw_infos.values():
            if isinstance(v, list):
                merged.extend(v)
        if merged:
            return merged
    raise TypeError(f"Unsupported infos format: {type(raw_infos)}")


def _build_sensor_entry(nusc: NuScenes, sample: Dict[str, Any], sensor_name: str) -> Dict[str, Any]:
    sd = nusc.get("sample_data", sample["data"][sensor_name])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])

    entry = {
        "filename": sd["filename"],
        "calib": {
            "rotation": cs["rotation"],
            "translation": cs["translation"],
        },
        "pose": {
            "rotation": ep["rotation"],
            "translation": ep["translation"],
        },
    }
    if sensor_name in CAMERAS:
        entry["calib"]["camera_intrinsic"] = cs.get("camera_intrinsic", [])
    return entry


def _ensure_required_arrays(info: Dict[str, Any]) -> None:
    gt_boxes = _as_numpy(info.get("gt_boxes", _zeros((0, 7), np.float32)), np.float32)
    n = int(gt_boxes.shape[0])

    gt_names = info.get("gt_names", np.array([], dtype=object))
    gt_names = np.asarray(gt_names)
    if gt_names.shape[0] != n:
        gt_names = np.array(["car"] * n)

    gt_velocity = _as_numpy(info.get("gt_velocity", _zeros((n, 2), np.float32)), np.float32)
    if gt_velocity.shape != (n, 2):
        gt_velocity = _zeros((n, 2), np.float32)

    num_lidar_pts = _as_numpy(info.get("num_lidar_pts", np.ones((n,), dtype=np.int32)), np.int32)
    if num_lidar_pts.shape[0] != n:
        num_lidar_pts = np.ones((n,), dtype=np.int32)

    num_radar_pts = _as_numpy(info.get("num_radar_pts", np.zeros((n,), dtype=np.int32)), np.int32)
    if num_radar_pts.shape[0] != n:
        num_radar_pts = np.zeros((n,), dtype=np.int32)

    info["gt_boxes"] = gt_boxes
    info["gt_names"] = gt_names
    info["gt_velocity"] = gt_velocity
    info["num_lidar_pts"] = num_lidar_pts
    info["num_radar_pts"] = num_radar_pts

    if "gt_map" not in info or not isinstance(info["gt_map"], dict):
        info["gt_map"] = _empty_gt_map()
    else:
        for k in ("divider", "ped_crossing", "boundary"):
            info["gt_map"].setdefault(k, [])

    info.setdefault("sweeps", [])
    info.setdefault("fut_valid_flag", np.array(False, dtype=np.bool_))

    # Ego future labels (minimal placeholders compatible with downstream shapes).
    info.setdefault("gt_ego_his_trajs", _zeros((HISTORY_STEPS, 2), np.float32))
    info.setdefault("gt_ego_fut_trajs", _zeros((FUTURE_STEPS, 2), np.float32))
    info.setdefault("gt_ego_fut_masks", _zeros((FUTURE_STEPS,), np.float32))
    info.setdefault("gt_ego_fut_cmd", DEFAULT_EGO_FUT_CMD.copy())
    info.setdefault("gt_ego_lcf_feat", _zeros((9,), np.float32))

    # Agent future labels for planner metric.
    info.setdefault("gt_agent_fut_trajs", _zeros((n, FUTURE_STEPS * 2), np.float32))
    info.setdefault("gt_agent_fut_masks", _zeros((n, FUTURE_STEPS), np.float32))
    info.setdefault("gt_agent_fut_goal", _zeros((n,), np.float32))

    # lcf_feat index 27 stores type in metric_stp3; keep 9 dims with zeros by default.
    info.setdefault("gt_agent_lcf_feat", _zeros((n, 9), np.float32))
    info.setdefault("gt_agent_fut_yaw", _zeros((n, FUTURE_STEPS), np.float32))

    # Normalize shapes/dtypes again if source has malformed values.
    info["gt_ego_his_trajs"] = _as_numpy(info["gt_ego_his_trajs"], np.float32).reshape(-1, 2)
    info["gt_ego_fut_trajs"] = _as_numpy(info["gt_ego_fut_trajs"], np.float32).reshape(-1, 2)
    info["gt_ego_fut_masks"] = _as_numpy(info["gt_ego_fut_masks"], np.float32).reshape(-1)
    info["gt_ego_fut_cmd"] = _normalize_ego_fut_cmd(info["gt_ego_fut_cmd"])
    info["gt_ego_lcf_feat"] = _as_numpy(info["gt_ego_lcf_feat"], np.float32).reshape(-1)

    info["gt_agent_fut_trajs"] = _as_numpy(info["gt_agent_fut_trajs"], np.float32).reshape(n, -1) if n > 0 else _zeros((0, FUTURE_STEPS * 2), np.float32)
    if n > 0 and (info["gt_agent_fut_trajs"].shape[1] != FUTURE_STEPS * 2 or not np.any(info["gt_agent_fut_trajs"])):
        info["gt_agent_fut_trajs"] = _zeros((n, FUTURE_STEPS * 2), np.float32)

    info["gt_agent_fut_masks"] = _as_numpy(info["gt_agent_fut_masks"], np.float32).reshape(n, -1) if n > 0 else _zeros((0, FUTURE_STEPS), np.float32)
    if n > 0 and (info["gt_agent_fut_masks"].shape[1] != FUTURE_STEPS or not np.any(info["gt_agent_fut_masks"])):
        info["gt_agent_fut_masks"] = _zeros((n, FUTURE_STEPS), np.float32)

    info["gt_agent_fut_goal"] = _as_numpy(info["gt_agent_fut_goal"], np.float32).reshape(n) if n > 0 else _zeros((0,), np.float32)
    info["gt_agent_lcf_feat"] = _build_agent_lcf_feat(gt_boxes, gt_velocity, gt_names, info["gt_agent_lcf_feat"])
    info["gt_agent_fut_yaw"] = _as_numpy(info["gt_agent_fut_yaw"], np.float32).reshape(n, -1) if n > 0 else _zeros((0, FUTURE_STEPS), np.float32)
    if n > 0 and info["gt_agent_fut_yaw"].shape[1] != FUTURE_STEPS:
        info["gt_agent_fut_yaw"] = _zeros((n, FUTURE_STEPS), np.float32)


def _convert_one(
    src_path: str,
    dst_path: str,
    nusc: NuScenes,
    occ_dir: Optional[str],
    map_pc_range: Tuple[float, float, float, float, float, float],
    plan_valid_pc_range: Tuple[float, float, float, float, float, float],
    min_map_line_length: float,
    map_simplify_tolerance: float,
    zero_incomplete_plan: bool,
    mask_plan_outside_range: bool,
    strict: bool,
    skip_missing_occ: bool,
    max_samples: int,
) -> None:
    src = mmengine.load(src_path)
    if not isinstance(src, dict) or "infos" not in src:
        raise ValueError(f"{src_path} must contain top-level key 'infos'.")

    infos = _get_list_infos(src["infos"])
    if max_samples > 0:
        infos = infos[:max_samples]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    skipped_no_token = 0
    skipped_strict = 0
    skipped_missing_occ = 0
    raw_progress = tqdm(infos, desc=f"convert:{os.path.basename(src_path)}", unit="frame")
    map_builder = _MapAnnotationBuilder(
        nusc.dataroot,
        patch_size=(float(map_pc_range[4] - map_pc_range[1]), float(map_pc_range[3] - map_pc_range[0])),
        min_line_length=min_map_line_length,
        simplify_tolerance=map_simplify_tolerance,
    )

    for raw in raw_progress:
        info = deepcopy(raw)
        token = _get_token(info)
        if token is None:
            skipped_no_token += 1
            raw_progress.set_postfix(skipped_no_token=skipped_no_token, skipped_missing_occ=skipped_missing_occ)
            continue

        sample = nusc.get("sample", token)
        scene_token = sample["scene_token"]

        # Rebuild data tree from nuScenes DB so geometry fields are valid.
        info["data"] = {}
        for s in SENSORS:
            info["data"][s] = _build_sensor_entry(nusc, sample, s)

        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        scene = nusc.get("scene", scene_token)
        log = nusc.get("log", scene["log_token"])

        info["token"] = token
        info["timestamp"] = info.get("timestamp", sample["timestamp"])
        info["lidar_path"] = info.get("lidar_path", lidar_sd["filename"])

        resolved_occ_path = _resolve_surroundocc_path(info["lidar_path"], occ_dir)
        if occ_dir and resolved_occ_path is None and skip_missing_occ:
            skipped_missing_occ += 1
            raw_progress.set_postfix(skipped_no_token=skipped_no_token, skipped_missing_occ=skipped_missing_occ)
            continue

        info["occ_path"] = resolved_occ_path or info.get("occ_path", info["lidar_path"])
        info["has_surroundocc"] = np.array(resolved_occ_path is not None, dtype=np.bool_)

        info["ego2global_rotation"] = info.get("ego2global_rotation", lidar_ep["rotation"])
        info["ego2global_translation"] = info.get("ego2global_translation", lidar_ep["translation"])
        info["lidar2ego_rotation"] = info.get("lidar2ego_rotation", lidar_cs["rotation"])
        info["lidar2ego_translation"] = info.get("lidar2ego_translation", lidar_cs["translation"])
        info["map_location"] = info.get("map_location", log["location"])

        before_keys = set(info.keys())
        _ensure_required_arrays(info)
        after_keys = set(info.keys())

        if strict:
            must_exist = [
                "gt_map",
                "gt_ego_fut_trajs",
                "gt_ego_fut_masks",
                "gt_ego_fut_cmd",
                "gt_agent_fut_trajs",
                "gt_agent_fut_masks",
                "gt_agent_fut_goal",
                "gt_agent_lcf_feat",
                "gt_agent_fut_yaw",
            ]
            added = [k for k in must_exist if k not in before_keys and k in after_keys]
            if added:
                skipped_strict += 1
                raw_progress.set_postfix(skipped_no_token=skipped_no_token, skipped_missing_occ=skipped_missing_occ, skipped_strict=skipped_strict)
                continue

        grouped[scene_token].append(info)
        if len(grouped) % 50 == 0:
            raw_progress.set_postfix(scenes=len(grouped), skipped_no_token=skipped_no_token, skipped_missing_occ=skipped_missing_occ, skipped_strict=skipped_strict)

    raw_progress.close()

    # Sort each scene by timestamp and produce metadata index.
    metadata: List[Tuple[str, int]] = []
    map_supervision_frames = 0
    agent_future_frames = 0
    full_ego_future_frames = 0
    scene_items = list(grouped.items())
    scene_progress = tqdm(scene_items, desc=f"enrich:{os.path.basename(src_path)}", unit="scene")
    for scene_token, arr in scene_progress:
        arr.sort(key=lambda x: x["timestamp"])
        for info in arr:
            info["gt_map"] = _sanitize_gt_map(
                map_builder.build_for_info(info),
                map_pc_range,
                min_map_line_length,
                map_simplify_tolerance,
            )
        _fill_scene_agent_future_labels(arr, nusc)
        _fill_scene_planner_labels(
            arr,
            plan_valid_pc_range=plan_valid_pc_range,
            zero_incomplete_plan=zero_incomplete_plan,
            mask_plan_outside_range=mask_plan_outside_range,
        )
        for idx in range(len(arr)):
            if any(len(v) > 0 for v in arr[idx]["gt_map"].values()):
                map_supervision_frames += 1
            if np.any(arr[idx]["gt_agent_fut_masks"]):
                agent_future_frames += 1
            if bool(_as_numpy(arr[idx]["fut_valid_flag"]).reshape(-1)[0]):
                full_ego_future_frames += 1
            metadata.append((scene_token, idx))
        scene_progress.set_postfix(
            keyframes=len(metadata),
            map_frames=map_supervision_frames,
            agent_future_frames=agent_future_frames,
            full_ego_future_frames=full_ego_future_frames,
        )

    scene_progress.close()

    metadata.sort(key=lambda x: x[0] + f"{x[1]:03d}")

    out = {
        "infos": dict(grouped),
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mmengine.dump(out, dst_path)

    print(f"[OK] wrote: {dst_path}")
    print(
        "[STAT] "
        f"scenes={len(out['infos'])}, keyframes={len(out['metadata'])}, "
        f"skip_no_token={skipped_no_token}, skip_strict={skipped_strict}, "
        f"skip_missing_occ={skipped_missing_occ}, map_supervision_frames={map_supervision_frames}, "
        f"agent_future_frames={agent_future_frames}, full_ego_future_frames={full_ego_future_frames}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generic nuScenes infos to GaussianAD infos.")
    parser.add_argument("--dataroot", default="data/nuscenes", help="nuScenes dataroot")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")

    parser.add_argument("--src-train", default="data/nuscenes_cam/nuscenes_infos_train.pkl")
    parser.add_argument("--src-val", default="data/nuscenes_cam/nuscenes_infos_val.pkl")

    parser.add_argument("--dst-train", default="data/nuscenes_cam/nuscenes_infos_train_gaussian_ad.pkl")
    parser.add_argument("--dst-val", default="data/nuscenes_cam/nuscenes_infos_val_gaussian_ad.pkl")
    parser.add_argument("--surroundocc-train-dir", default="data/surroundocc/train_samples")
    parser.add_argument("--surroundocc-val-dir", default="data/surroundocc/val_samples")
    parser.add_argument(
        "--map-pc-range",
        type=float,
        nargs=6,
        default=DEFAULT_MAP_PC_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Point cloud range used to generate local map supervision for gt_map.",
    )
    parser.add_argument(
        "--plan-valid-pc-range",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help=(
            "Range used to mask ego future steps. Defaults to --map-pc-range. "
            "This is useful for convergence when long future points leave the BEV/map supervision range."
        ),
    )
    parser.add_argument(
        "--min-map-line-length",
        type=float,
        default=DEFAULT_MIN_MAP_LINE_LENGTH,
        help="Drop local map line fragments shorter than this many meters.",
    )
    parser.add_argument(
        "--map-simplify-tolerance",
        type=float,
        default=DEFAULT_MAP_SIMPLIFY_TOLERANCE,
        help="Douglas-Peucker tolerance in meters for local map line cleanup. Set 0 to disable.",
    )
    parser.add_argument(
        "--zero-incomplete-plan",
        action="store_true",
        help="Set ego future masks/trajs to zero for frames that do not have all six future steps.",
    )
    parser.add_argument(
        "--mask-plan-outside-range",
        action="store_true",
        help="Mask ego future steps after the cumulative trajectory leaves --plan-valid-pc-range.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "If set, skip samples that miss planning/map keys in source infos, "
            "instead of auto-filling placeholders."
        ),
    )
    parser.add_argument(
        "--skip-missing-occ",
        action="store_true",
        help="If set, skip samples that do not have a matching SurroundOcc .npy file.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Do not convert the training split.")
    parser.add_argument("--skip-val", action="store_true", help="Do not convert the validation split.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug only: convert at most this many train frames.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Debug only: convert at most this many val frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    map_pc_range = tuple(args.map_pc_range)
    plan_valid_pc_range = tuple(args.plan_valid_pc_range) if args.plan_valid_pc_range is not None else map_pc_range

    if not args.skip_train:
        _convert_one(
            args.src_train,
            args.dst_train,
            nusc,
            occ_dir=args.surroundocc_train_dir,
            map_pc_range=map_pc_range,
            plan_valid_pc_range=plan_valid_pc_range,
            min_map_line_length=args.min_map_line_length,
            map_simplify_tolerance=args.map_simplify_tolerance,
            zero_incomplete_plan=args.zero_incomplete_plan,
            mask_plan_outside_range=args.mask_plan_outside_range,
            strict=args.strict,
            skip_missing_occ=args.skip_missing_occ,
            max_samples=args.max_train_samples,
        )
    if not args.skip_val:
        _convert_one(
            args.src_val,
            args.dst_val,
            nusc,
            occ_dir=args.surroundocc_val_dir,
            map_pc_range=map_pc_range,
            plan_valid_pc_range=plan_valid_pc_range,
            min_map_line_length=args.min_map_line_length,
            map_simplify_tolerance=args.map_simplify_tolerance,
            zero_incomplete_plan=args.zero_incomplete_plan,
            mask_plan_outside_range=args.mask_plan_outside_range,
            strict=args.strict,
            skip_missing_occ=args.skip_missing_occ,
            max_samples=args.max_val_samples,
        )

    print("[DONE] conversion finished.")


if __name__ == "__main__":
    main()
