from __future__ import annotations

import unittest
from datetime import date

from tgsi_pipeline.models import DateRange, Location
from tgsi_pipeline.sources.remote_sensing import (
    LayerSpec,
    _parse_point_csv,
    _pick_ndvi_layers,
    _pick_soil_moisture_layers,
)


class RemoteSensingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.location = Location(
            id="sorriso_mt",
            name="Sorriso, MT",
            latitude=-12.5425,
            longitude=-55.7211,
        )
        self.date_range = DateRange(
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )

    def test_pick_ndvi_layer_prefers_main_ndvi(self) -> None:
        product = {
            "DetailedQA": {},
            "NDVI": {},
            "SummaryQA": {},
        }
        self.assertEqual(_pick_ndvi_layers(product), ["NDVI"])

    def test_pick_soil_layers_prefers_am_and_pm(self) -> None:
        product = {
            "retrieval_qual_flag_pm": {},
            "soil_moisture_pm": {},
            "soil_moisture_am": {},
        }
        self.assertEqual(
            _pick_soil_moisture_layers(product),
            ["soil_moisture_am", "soil_moisture_pm"],
        )

    def test_parse_point_csv_scales_unscaled_ndvi(self) -> None:
        csv_text = "Date,MOD13Q1_061_NDVI\n2024-01-17,8432\n"
        rows = _parse_point_csv(
            csv_text,
            location=self.location,
            date_range=self.date_range,
            specs=[
                LayerSpec(
                    product_id="MOD13Q1.061",
                    layer_name="NDVI",
                    value_field="ndvi",
                    source_field="ndvi_source",
                    scale_factor=0.0001,
                    add_offset=0.0,
                    fill_value=-3000,
                    valid_min=-2000,
                    valid_max=10000,
                )
            ],
            value_field="ndvi",
            source_field="ndvi_source",
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["ndvi"], 0.8432, places=6)

    def test_parse_point_csv_keeps_scaled_soil_moisture(self) -> None:
        csv_text = "Date,SPL3SMP_E_006_soil_moisture_am,SPL3SMP_E_006_soil_moisture_pm\n2024-01-02,0.22,0.26\n"
        rows = _parse_point_csv(
            csv_text,
            location=self.location,
            date_range=self.date_range,
            specs=[
                LayerSpec(
                    product_id="SPL3SMP_E.006",
                    layer_name="soil_moisture_am",
                    value_field="soil_moisture_m3m3",
                    source_field="soil_moisture_source",
                    scale_factor=0.001,
                    add_offset=0.0,
                    fill_value=-9999,
                    valid_min=0.0,
                    valid_max=5000,
                ),
                LayerSpec(
                    product_id="SPL3SMP_E.006",
                    layer_name="soil_moisture_pm",
                    value_field="soil_moisture_m3m3",
                    source_field="soil_moisture_source",
                    scale_factor=0.001,
                    add_offset=0.0,
                    fill_value=-9999,
                    valid_min=0.0,
                    valid_max=5000,
                ),
            ],
            value_field="soil_moisture_m3m3",
            source_field="soil_moisture_source",
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["soil_moisture_m3m3"], 0.24, places=6)


if __name__ == "__main__":
    unittest.main()
