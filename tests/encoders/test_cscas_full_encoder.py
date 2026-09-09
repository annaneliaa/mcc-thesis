from __future__ import annotations

from thesis.encoders.cscas_full import (
    CSCAS_FULL_FEATURES,
    compute_cscas_full_features,
)
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS


def _ag(**overrides) -> AlertGroup:
    defaults = dict(
        alert_group_id="g1",
        group_id="g1",
        method="cscas_pregrouped",
        start_ts=1_642_636_800,
        end_ts=1_642_636_800,
        n_alerts=3,
        proto=6,
        ext_port=443,
        int_port=51000,
        category="EXPLOIT",
        ruleset="ET",
        signature_matches_per_day=1234.5,
        similarity=0.91,
        signature_id_similarity=0.88,
        scas=1,
        attr_similarities={"AppProtoSimilarity": 0.7},
    )
    return AlertGroup(**{**defaults, **overrides})


def test_feature_list_shape():
    # 5 base + similarity + signature_id_similarity + scas + one per attr col
    assert len(CSCAS_FULL_FEATURES) == 8 + len(ATTR_SIMILARITY_COLUMNS)
    assert CSCAS_FULL_FEATURES[:5] == [
        "proto",
        "ext_port",
        "int_port",
        "n_alerts",
        "signature_matches_per_day",
    ]
    assert "scas" in CSCAS_FULL_FEATURES


def test_compute_pulls_offline_scores_and_fills_missing():
    feats = compute_cscas_full_features(_ag())
    assert feats["similarity"] == 0.91
    assert feats["signature_id_similarity"] == 0.88
    assert feats["scas"] == 1.0
    assert feats["attr_value:AppProtoSimilarity"] == 0.7
    # an unpopulated *Similarity column -> the -1.0 not-applicable sentinel
    assert feats["attr_value:DnsRrnameSimilarity"] == -1.0

    missing = compute_cscas_full_features(
        _ag(
            similarity=None,
            signature_id_similarity=None,
            scas=None,
            attr_similarities=None,
        )
    )
    assert missing["similarity"] == 0.0
    assert missing["signature_id_similarity"] == 0.0
    assert missing["scas"] == -1.0


def test_encoder_service_routes_on_kind_and_selects_schema_columns():
    rows = [_ag(), _ag(alert_group_id="g2", scas=0)]
    schema = FeatureSchema(
        schema_name="cscas_full",
        schema_version="0.1.0",
        base=BaseFeatureSchema(CSCAS_FULL_FEATURES, kind="cscas_full"),
        symbolic=None,
    )
    X = encode_alert_groups_for_schema(rows, schema)
    assert list(X.columns) == CSCAS_FULL_FEATURES
    assert len(X) == 2

    # a schema that omits scas (what one-class models get) yields no scas column
    no_scas = [f for f in CSCAS_FULL_FEATURES if f != "scas"]
    schema_no_scas = FeatureSchema(
        schema_name="cscas_full",
        schema_version="0.1.0",
        base=BaseFeatureSchema(no_scas, kind="cscas_full"),
        symbolic=None,
    )
    X2 = encode_alert_groups_for_schema(rows, schema_no_scas)
    assert "scas" not in X2.columns
    assert list(X2.columns) == no_scas


def test_default_kind_still_uses_the_baseline_encoder():
    # a plain base schema (kind defaults to "baseline") must not gain the
    # cscas_full columns
    schema = FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(
            ["proto", "ext_port", "int_port", "n_alerts", "signature_matches_per_day"]
        ),
        symbolic=None,
    )
    X = encode_alert_groups_for_schema([_ag()], schema)
    assert "similarity" not in X.columns
    assert "scas" not in X.columns
