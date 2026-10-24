"""Load evaluation artifacts."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from cd2r.model.dual_channel_gnn import DualChannelGNN
from cd2r.model.entity_mapping import remap_kg_triplets_to_model_space
from cd2r.model.single_channel_gnn import DualChannelGNN as SingleChannelGNN

def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _read_pairs(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs

@dataclass
class EvalArtifacts:
    dataset: str
    variant: str
    artifact_dir: str
    meta: dict
    patient2local: Dict[str, int]
    doctor2local: Dict[str, int]
    train_user_list: dict
    n_users: int
    n_items: int
    pop_info: dict
    dr_info_local: Dict[int, dict]
    q_illness_primary: Dict[int, str]
    patient_query_illness: Dict[int, set]
    patient_feat: torch.Tensor
    intent_ids: torch.Tensor
    user_pop_id_map: Dict[int, int]
    item_pop_id_map: Dict[int, int]
    doctor_degrees: Dict[str, int]
    test_pairs: List[Tuple[str, str]]
    model: torch.nn.Module
    device: str

def _build_entity_maps(id2global: Dict[str, int]):
    patient_ids, doctor_ids = [], []
    for key, gid in sorted(id2global.items(), key=lambda kv: int(kv[1])):
        if key.startswith("patients_"):
            patient_ids.append(key[len("patients_") :])
        elif key.startswith("doctors_"):
            doctor_ids.append(key[len("doctors_") :])
    return (
        {raw: i for i, raw in enumerate(patient_ids)},
        {raw: i for i, raw in enumerate(doctor_ids)},
    )

def _load_train_lists(train_path, patient2local, doctor2local):
    train_user_list = defaultdict(list)
    with open(train_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            uid, did = parts[0], parts[1]
            if uid not in patient2local or did not in doctor2local:
                continue
            train_user_list[patient2local[uid]].append(doctor2local[did])
    return dict(train_user_list)

def load_artifacts(artifact_dir: str, device: str = "cpu") -> EvalArtifacts:
    meta = _load_json(os.path.join(artifact_dir, "meta.json"))
    gm = _load_json(os.path.join(artifact_dir, "global_id_mapping.json"))
    id2global = gm["id2global"]
    patient2local, doctor2local = _build_entity_maps(id2global)
    n_users, n_items = len(patient2local), len(doctor2local)

    pop_info = _load_json(os.path.join(artifact_dir, "pop_info.json"))
    train_user_list = _load_train_lists(
        os.path.join(artifact_dir, "train.txt"), patient2local, doctor2local
    )
    test_pairs = _read_pairs(os.path.join(artifact_dir, "test.txt"))
    doctor_degrees = {
        str(k): int(v)
        for k, v in _load_json(os.path.join(artifact_dir, "doctor_degrees.json")).items()
    }

    q_primary_raw = _load_json(os.path.join(artifact_dir, "q_illness_primary.json"))
    q_illness_primary = {int(k): str(v) for k, v in q_primary_raw.items()}

    pq_raw = _load_json(os.path.join(artifact_dir, "patient_query_illness.json"))
    patient_query_illness = {int(k): set(map(str, v)) for k, v in pq_raw.items()}

    side = torch.load(os.path.join(artifact_dir, "patient_side.pt"), map_location="cpu")
    patient_feat = side["patient_feat"].float()
    intent_ids = side["intent_ids"].long()

    content = torch.load(os.path.join(artifact_dir, "content_embs.pt"), map_location="cpu")
    aux = {
        "user_content": content["patient_content_matrix"].float(),
        "item_content": content["doctor_content_matrix"].float(),
        "user_proto": content["patient_proto_matrix"].float(),
        "item_proto": content["doctor_proto_matrix"].float(),
        "user_degree": torch.tensor(
            [len(train_user_list.get(u, [])) for u in range(n_users)], dtype=torch.float32
        ),
        "item_degree": torch.zeros(n_items, dtype=torch.float32),
    }
    for u, items in train_user_list.items():
        for i in items:
            aux["item_degree"][i] += 1

    dr_raw = _load_json(os.path.join(artifact_dir, "dr_info.json"))
    dr_info_local = {}
    for raw_id, info in dr_raw.items():
        if raw_id in doctor2local:
            dr_info_local[doctor2local[raw_id]] = {
                "illness_ids": [str(x) for x in info.get("illness_ids", [])],
                "illness_cnt": info.get("illness_cnt", []),
            }

    user_pop_id_map = {
        patient2local[uid]: int(pid)
        for uid, pid in pop_info.get("user_pop_idx", {}).items()
        if uid in patient2local
    }
    item_pop_id_map = {
        doctor2local[iid]: int(pid)
        for iid, pid in pop_info.get("item_pop_idx", {}).items()
        if iid in doctor2local and doctor2local[iid] < n_items
    }

    kg_path = os.path.join(artifact_dir, "maitai.kg")
    ModelCls = SingleChannelGNN if meta.get("single_channel") else DualChannelGNN
    kg_raw = DualChannelGNN.load_kg_triplets(kg_path)
    kg_triplets, n_total_entities, _ = remap_kg_triplets_to_model_space(
        kg_raw, id2global, patient2local, doctor2local, n_users, n_items
    )

    model_kwargs = dict(
        n_users=n_users,
        n_items=n_items,
        n_user_pop=pop_info["n_user_pop"],
        n_item_pop=pop_info["n_item_pop"],
        n_total_entities=n_total_entities,
        emb_dim=int(meta.get("emb_dim", 64)),
        n_layers=int(meta.get("n_layers", 2)),
    )
    model = ModelCls(**model_kwargs).to(device)
    model.set_auxiliary_features(aux)
    model.build_graphs(train_user_list, kg_triplets, device)
    state = torch.load(os.path.join(artifact_dir, "model.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    return EvalArtifacts(
        dataset=meta.get("dataset", "mixed"),
        variant=meta.get("variant", "full"),
        artifact_dir=artifact_dir,
        meta=meta,
        patient2local=patient2local,
        doctor2local=doctor2local,
        train_user_list=train_user_list,
        n_users=n_users,
        n_items=n_items,
        pop_info=pop_info,
        dr_info_local=dr_info_local,
        q_illness_primary=q_illness_primary,
        patient_query_illness=patient_query_illness,
        patient_feat=patient_feat,
        intent_ids=intent_ids,
        user_pop_id_map=user_pop_id_map,
        item_pop_id_map=item_pop_id_map,
        doctor_degrees=doctor_degrees,
        test_pairs=test_pairs,
        model=model,
        device=device,
    )
